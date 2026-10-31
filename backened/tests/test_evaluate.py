import pandas as pd
import pytest

import evaluate as ev

DF = pd.DataFrame({
    "Region": ["West", "West", "East", "South"],
    "Sales": [100.0, 200.0, 250.0, 50.0],
})


def resp(**kw):
    base = {"status": "success", "result_type": "number", "value": None, "rows": None, "error": None, "message": None}
    base.update(kw)
    return base


SCALAR = {"id": "t", "kind": "scalar", "expected": "df['Sales'].sum()"}
TABLE = {"id": "t", "kind": "table", "expected": "df.groupby('Region')['Sales'].sum()"}


def test_scalar_pass_and_fail():
    assert ev.check_case(SCALAR, resp(value=600.0), DF)[0]
    ok, why = ev.check_case(SCALAR, resp(value=650.0), DF)
    assert not ok and "got 650.00, expected 600.00" in why


def test_scalar_rejects_table_answer():
    ok, why = ev.check_case(SCALAR, resp(result_type="table", rows=[["West", 1.0]]), DF)
    assert not ok and "single number" in why


def test_table_pass_ignores_row_order():
    rows = [["South", 50.0], ["East", 250.0], ["West", 300.0]]
    assert ev.check_case(TABLE, resp(result_type="table", rows=rows), DF)[0]


def test_table_detects_wrong_value():
    rows = [["South", 50.0], ["East", 250.0], ["West", 999.0]]
    ok, why = ev.check_case(TABLE, resp(result_type="table", rows=rows), DF)
    assert not ok and "wrong value" in why


def test_table_detects_missing_and_extra_rows():
    ok, why = ev.check_case(TABLE, resp(result_type="table", rows=[["West", 300.0], ["North", 1.0]]), DF)
    assert not ok and "row mismatch" in why


def test_chart_requirement():
    case = {**TABLE, "chart": True}
    rows = [["South", 50.0], ["East", 250.0], ["West", 300.0]]
    assert ev.check_case(case, resp(result_type="chart", rows=rows), DF)[0]
    ok, why = ev.check_case(case, resp(result_type="table", rows=rows), DF)
    assert not ok and "chart" in why


def test_clarification_cases():
    case = {"id": "c", "kind": "clarification"}
    assert ev.check_case(case, resp(status="clarification"), None)[0]
    assert ev.check_case(case, resp(status="error"), None)[0]
    assert not ev.check_case(case, resp(status="success", value=1), None)[0]


def test_error_status_fails_answerable_question():
    ok, why = ev.check_case(SCALAR, resp(status="error", error="Ollama down"), DF)
    assert not ok and "Ollama down" in why


def test_multi_key_groupby_keys_match():
    df = pd.DataFrame({"a": ["x", "x"], "b": ["p", "q"], "v": [1.0, 2.0]})
    case = {"id": "m", "kind": "table", "expected": "df.groupby(['a', 'b'])['v'].sum()"}
    assert ev.check_case(case, resp(result_type="table", rows=[["x", "p", 1.0], ["x", "q", 2.0]]), df)[0]


def test_expression_cannot_use_builtins():
    with pytest.raises(NameError):
        ev.compute_expected("__import__('os').getcwd()", DF)


def test_question_file_is_well_formed():
    cases = ev.load_cases()
    assert len(cases) >= 30
    assert {c["dataset"] for c in cases} == {"superstore", "students", "hr_attrition"}
    assert any(c["kind"] == "clarification" for c in cases)


def test_expression_can_use_str_for_period_labels():
    df = pd.DataFrame({"d": pd.to_datetime(["2016-01-05", "2016-01-20", "2016-02-01"]), "v": [1.0, 2.0, 3.0]})
    exp = ev.compute_expected("df.groupby(df['d'].dt.to_period('M').astype(str))['v'].sum()", df)
    assert ev.series_to_dict(exp) == {("2016-01",): 3.0, ("2016-02",): 3.0}
