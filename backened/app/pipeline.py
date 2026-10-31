"""Pipeline: question -> LLM plan -> validate -> run -> format.

Always returns a dict with the same keys, so the frontend never has to guess.
status is "success", "clarification" or "error".
"""
import logging

from app import dataset_manager as dm
from app.engine import EngineError, run_plan
from app.formatter import format_response
from app.llm_service import LLMError, ask_llm
from app.repair import repair_plan
from app.validator import ValidationError, validate_plan

log = logging.getLogger("ai_data_analyst")


def _base(dataset, question) -> dict:
    return {
        "status": None, "dataset": dataset, "question": question,
        "result_type": None, "label": None, "value": None,
        "columns": None, "rows": None, "row_count": None, "truncated": None,
        "chart": None, "explanation": None,
        "message": None,   # filled for status == "clarification"
        "error": None,     # filled for status == "error"
        "plan": None,      # the validated plan, for transparency
    }


def answer_question(dataset: str, question: str, client=None) -> dict:
    out = _base(dataset, question)
    try:
        df = dm.load_dataset(dataset)
        plan = ask_llm(dataset, question, client=client)

        if "clarification_needed" in plan:
            out.update(status="clarification", message=plan["clarification_needed"])
            return out

        clean = validate_plan(plan, df, dm.get_aliases(dataset))
        clean = repair_plan(clean, question, df)
        result = run_plan(df, clean)
        out.update(format_response(result, clean))
        out.update(status="success", plan=clean)
    except (dm.DatasetError, LLMError, ValidationError, EngineError) as e:
        out.update(status="error", error=str(e))
    except Exception:  # never leak internals to the user
        log.exception("Unexpected error while answering a question")
        out.update(status="error", error="Something went wrong while analysing that question. Please try again.")
    return out