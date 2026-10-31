import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import dataset_manager as dm
from app import pipeline
from app.llm_service import LLMError
from app.main import app

DF = pd.DataFrame({
    "Region": ["West", "West", "East", "South"],
    "Sales": [100.0, 200.0, 250.0, 50.0],
})
KEYS = {"status", "dataset", "question", "result_type", "label", "value", "columns", "rows",
        "row_count", "truncated", "chart", "explanation", "message", "error", "plan"}


@pytest.fixture(autouse=True)
def fake_dataset(monkeypatch):
    def load(name):
        if name != "demo":
            raise dm.DatasetError(f"Unknown dataset '{name}'.")
        return DF
    monkeypatch.setattr(dm, "load_dataset", load)


def plan_returning(plan):
    return lambda dataset, question, client=None: plan


GOOD_PLAN = {"operation": "groupby_aggregate", "group_by": "region", "column": "sales",
             "aggregation": "sum", "sort": "desc", "output": "chart", "chart_type": "bar"}


def test_success_chart(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(GOOD_PLAN))
    r = pipeline.answer_question("demo", "sales by region")
    assert set(r) == KEYS and r["status"] == "success" and r["result_type"] == "chart"
    assert r["rows"][0] == ["West", 300.0]
    assert r["plan"]["group_by"] == ["Region"]       # column names repaired by validator
    assert r["error"] is None and r["message"] is None


def test_success_number(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(
        {"operation": "aggregate", "column": "Sales", "aggregation": "sum"}))
    r = pipeline.answer_question("demo", "total sales")
    assert r["result_type"] == "number" and r["value"] == 600.0


def test_clarification(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning({"clarification_needed": "No such column."}))
    r = pipeline.answer_question("demo", "salary?")
    assert r["status"] == "clarification" and r["message"] == "No such column." and r["error"] is None


def test_validation_error_is_reported(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(
        {"operation": "aggregate", "column": "Salary", "aggregation": "sum"}))
    r = pipeline.answer_question("demo", "q")
    assert r["status"] == "error" and "Salary" in r["error"] and r["result_type"] is None


def test_engine_error_is_reported(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(
        {"operation": "count_rows", "filters": [{"column": "Region", "op": "==", "value": "Mars"}]}))
    assert pipeline.answer_question("demo", "q")["status"] == "error"


def test_llm_error_is_reported(monkeypatch):
    def boom(dataset, question, client=None):
        raise LLMError("I can't reach Ollama.")
    monkeypatch.setattr(pipeline, "ask_llm", boom)
    r = pipeline.answer_question("demo", "q")
    assert r["status"] == "error" and "Ollama" in r["error"]


def test_unknown_dataset_is_reported_without_calling_llm(monkeypatch):
    def must_not_run(*a, **k):
        raise AssertionError("LLM should not be called")
    monkeypatch.setattr(pipeline, "ask_llm", must_not_run)
    assert pipeline.answer_question("nope", "q")["status"] == "error"


def test_unexpected_exception_is_hidden(monkeypatch):
    def boom(dataset, question, client=None):
        raise RuntimeError("secret internal detail")
    monkeypatch.setattr(pipeline, "ask_llm", boom)
    r = pipeline.answer_question("demo", "q")
    assert r["status"] == "error" and "secret" not in r["error"]


# ---------- HTTP layer ----------

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_datasets_endpoint(monkeypatch):
    monkeypatch.setattr(dm, "list_datasets", lambda: [{"name": "demo", "label": "Demo", "description": "d"}])
    assert client.get("/datasets").json()[0]["name"] == "demo"


def test_preview_404_for_unknown(monkeypatch):
    def missing(name):
        raise dm.DatasetError("Unknown dataset 'x'.")
    monkeypatch.setattr(dm, "get_schema", missing)
    assert client.get("/datasets/x/preview").status_code == 404


def test_preview_ok(monkeypatch):
    monkeypatch.setattr(dm, "get_schema", lambda name: {"name": name, "rows": 4})
    assert client.get("/datasets/demo/preview").json() == {"name": "demo", "rows": 4}


def test_ask_endpoint(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(GOOD_PLAN))
    r = client.post("/ask", json={"dataset": "demo", "question": "sales by region"})
    assert r.status_code == 200 and r.json()["status"] == "success"


def test_ask_rejects_malformed_body():
    assert client.post("/ask", json={"dataset": "demo"}).status_code == 422


def test_cors_header_present():
    r = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "*"


def test_pipeline_repairs_group_ranking_mistake(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(
        {"operation": "top_n", "sort_by": "Sales", "n": 2, "order": "desc", "columns": ["Region"]}))
    r = pipeline.answer_question("demo", "top 2 regions by total sales")
    assert r["status"] == "success" and r["plan"]["operation"] == "groupby_aggregate"
    assert r["rows"] == [["West", 300.0], ["East", 250.0]]


def test_pipeline_turns_named_group_into_filter(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(
        {"operation": "groupby_aggregate", "group_by": "Region", "column": "Sales", "aggregation": "mean"}))
    r = pipeline.answer_question("demo", "What is the average sales for the West region?")
    assert r["status"] == "success" and r["result_type"] == "number" and r["value"] == 150.0
    assert r["plan"]["operation"] == "aggregate"


def test_pipeline_collapses_group_filtered_to_one_value(monkeypatch):
    monkeypatch.setattr(pipeline, "ask_llm", plan_returning(
        {"operation": "groupby_aggregate", "group_by": ["Region"], "column": "Sales", "aggregation": "count",
         "filters": [{"column": "Region", "op": "==", "value": "West"}, {"column": "Sales", "op": ">", "value": 150}]}))
    r = pipeline.answer_question("demo", "How many sales in the West were above 150?")
    assert r["status"] == "success" and r["result_type"] == "number" and r["value"] == 1
    assert r["plan"]["operation"] == "aggregate"