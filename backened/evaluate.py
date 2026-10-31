"""Accuracy evaluation: runs every question in tests/test_questions.json through the
real pipeline and compares the answer with one computed directly in Pandas.

Run from backend/ (Ollama must be running):
    python evaluate.py                       # all questions
    python evaluate.py --dataset students    # one dataset
    python evaluate.py --ids hr-01 hr-02     # specific questions
    python evaluate.py --model qwen2.5:7b    # try another model
    python evaluate.py --save results.json   # keep the results

Question kinds:
    scalar         - one number; "expected" is a Pandas expression on df
    table          - grouped result; "expected" is a Pandas expression returning a Series
    clarification  - unanswerable question; passes if the app refuses instead of answering
"chart": true additionally requires the answer to be a chart.

"expected" expressions are written by you in the JSON file. They are never produced by the LLM.
"""
import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from app import dataset_manager as dm
from app import pipeline

QUESTIONS_PATH = Path(__file__).resolve().parent / "tests" / "test_questions.json"
TOLERANCE = 0.011  # engine rounds to 2 decimals
KINDS = {"scalar", "table", "clarification"}


# ---------- loading ----------

def load_cases(path=QUESTIONS_PATH) -> list[dict]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids in the question file.")
    for c in cases:
        if c["kind"] not in KINDS:
            raise ValueError(f"{c['id']}: kind must be one of {sorted(KINDS)}.")
        if c["kind"] != "clarification" and not c.get("expected"):
            raise ValueError(f"{c['id']}: missing 'expected' expression.")
    return cases


# ---------- comparing ----------

def compute_expected(expression: str, df: pd.DataFrame):
    return eval(expression, {"__builtins__": {}}, {"df": df, "pd": pd, "len": len, "str": str})


def _close(got, expected) -> bool:
    if got is None or expected is None:
        return got is None and expected is None
    try:
        return abs(float(got) - float(expected)) <= TOLERANCE
    except (TypeError, ValueError):
        return False


def _key(k) -> tuple:
    return tuple(str(x) for x in (k if isinstance(k, tuple) else (k,)))


def series_to_dict(series) -> dict:
    if not isinstance(series, pd.Series):
        raise ValueError("A table 'expected' expression must return a Pandas Series.")
    return {_key(k): v for k, v in series.items()}


def rows_to_dict(rows) -> dict:
    return {tuple(str(x) for x in r[:-1]): r[-1] for r in rows}


def _fmt(v) -> str:
    return f"{float(v):,.2f}" if isinstance(v, (int, float)) or hasattr(v, "item") else str(v)


def check_case(case: dict, response: dict, df: pd.DataFrame | None) -> tuple[bool, str]:
    """Return (passed, reason). reason is empty on success."""
    status = response["status"]
    kind = case["kind"]

    if kind == "clarification":
        if status == "success":
            return False, "answered a question that cannot be answered"
        return True, ""

    if status != "success":
        return False, f"status={status}: {response.get('error') or response.get('message')}"

    expected = compute_expected(case["expected"], df)
    rtype = response["result_type"]

    if kind == "scalar":
        if rtype != "number":
            return False, f"expected a single number, got a {rtype}"
        if not _close(response["value"], expected):
            return False, f"got {_fmt(response['value'])}, expected {_fmt(expected)}"
    else:
        if rtype not in ("table", "chart"):
            return False, f"expected a table, got a {rtype}"
        got, exp = rows_to_dict(response["rows"]), series_to_dict(expected)
        missing, extra = sorted(set(exp) - set(got)), sorted(set(got) - set(exp))
        if missing or extra:
            return False, (f"row mismatch (expected {len(exp)} rows, got {len(got)}); "
                           f"missing {missing[:3]}, unexpected {extra[:3]}")
        bad = [(k, got[k], exp[k]) for k in exp if not _close(got[k], exp[k])]
        if bad:
            k, g, e = bad[0]
            return False, f"{len(bad)} wrong value(s), e.g. {k}: got {_fmt(g)}, expected {_fmt(e)}"

    if case.get("chart") and rtype != "chart":
        return False, "expected a chart but got a " + str(rtype)
    return True, ""


# ---------- running ----------

def run(cases, save_path=None):
    frames, results = {}, []
    for i, case in enumerate(cases, 1):
        ds = case["dataset"]
        try:
            if ds not in frames:
                frames[ds] = dm.load_dataset(ds)
        except dm.DatasetError as e:
            print(f"SKIP  {case['id']:<7} {e}", flush=True)
            continue

        t0 = time.time()
        response = pipeline.answer_question(ds, case["question"])
        seconds = time.time() - t0
        try:
            ok, reason = check_case(case, response, frames[ds])
        except Exception as e:  # a bad expression in the question file
            ok, reason = False, f"evaluation error: {e}"

        print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<7} {seconds:5.1f}s  {case['question']}", flush=True)
        if not ok:
            print(f"        reason: {reason}")
            if response.get("plan"):
                print(f"        plan:   {json.dumps(response['plan'])}")
        results.append({"id": case["id"], "dataset": ds, "question": case["question"], "passed": ok,
                        "reason": reason, "seconds": round(seconds, 1), "plan": response.get("plan")})

    summarize(results)
    if save_path:
        Path(save_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults saved to {save_path}")
    return results


def summarize(results):
    if not results:
        print("\nNo questions were run.")
        return
    total, passed = len(results), sum(r["passed"] for r in results)
    print("\n" + "=" * 52)
    print(f"Accuracy: {passed}/{total} = {100 * passed / total:.0f}%")
    print(f"Average time per question: {sum(r['seconds'] for r in results) / total:.1f}s")
    for ds in sorted({r["dataset"] for r in results}):
        rs = [r for r in results if r["dataset"] == ds]
        print(f"  {ds:<14} {sum(r['passed'] for r in rs)}/{len(rs)}")
    failed = [r["id"] for r in results if not r["passed"]]
    if failed:
        print("Failed: " + ", ".join(failed))


def main():
    ap = argparse.ArgumentParser(description="Evaluate the AI Data Analyst against Pandas.")
    ap.add_argument("--dataset", help="only this dataset")
    ap.add_argument("--ids", nargs="+", help="only these question ids")
    ap.add_argument("--model", help="Ollama model to use (overrides OLLAMA_MODEL)")
    ap.add_argument("--save", metavar="FILE", help="write results to a JSON file")
    args = ap.parse_args()

    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model
    cases = load_cases()
    if args.dataset:
        cases = [c for c in cases if c["dataset"] == args.dataset]
    if args.ids:
        cases = [c for c in cases if c["id"] in args.ids]
    print(f"Model: {os.getenv('OLLAMA_MODEL') or 'default from llm_service'} | {len(cases)} questions\n", flush=True)
    run(cases, args.save)


if __name__ == "__main__":
    main()
