"""End-to-end check: question -> LLM plan -> validator -> engine.

Usage (from backend/):
    python try_llm.py superstore "total sales by region as a bar chart"
"""
import json
import sys

from app import dataset_manager as dm
from app.engine import EngineError, run_plan
from app.llm_service import LLMError, ask_llm
from app.validator import ValidationError, validate_plan

if len(sys.argv) < 3:
    sys.exit('Usage: python try_llm.py <dataset> "<question>"')

dataset, question = sys.argv[1], " ".join(sys.argv[2:])
try:
    df = dm.load_dataset(dataset)
    plan = ask_llm(dataset, question)
    print("LLM plan:\n", json.dumps(plan, indent=2))
    if "clarification_needed" in plan:
        sys.exit(0)
    clean = validate_plan(plan, df)
    print("\nValidated plan:\n", json.dumps(clean, indent=2, default=str))
    print("\nResult:\n", json.dumps(run_plan(df, clean), indent=2))
except (dm.DatasetError, LLMError, ValidationError, EngineError) as e:
    print("Error:", e)