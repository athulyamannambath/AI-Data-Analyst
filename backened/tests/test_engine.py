import pandas as pd
import pytest

from app.engine import EngineError, run_plan


@pytest.fixture
def df():
    return pd.DataFrame({
        "Region": ["West", "West", "East", "East", "South", "South"],
        "Category": ["Tech", "Furniture", "Tech", "Tech", "Furniture", "Tech"],
        "Sales": [100.0, 200.0, 300.0, 50.0, 400.0, None],
        "Profit": [10.0, -20.0, 30.0, 5.0, 40.0, 1.0],
        "Order Date": pd.to_datetime(
            ["2016-01-05", "2016-01-20", "2016-02-10", "2017-02-11", "2017-03-01", "2017-03-15"]),
    })


def test_count_rows_with_filter(df):
    r = run_plan(df, {"operation": "count_rows",
                      "filters": [{"column": "Region", "op": "==", "value": "west"}]})
    assert r["value"] == 2  # case-insensitive text match


def test_aggregate_sum(df):
    assert run_plan(df, {"operation": "aggregate", "column": "Sales",
                         "aggregation": "sum"})["value"] == 1050.0


def test_numeric_filter(df):
    r = run_plan(df, {"operation": "count_rows",
                      "filters": [{"column": "Profit", "op": ">", "value": 10}]})
    assert r["value"] == 2


def test_groupby_sort_limit(df):
    r = run_plan(df, {"operation": "groupby_aggregate", "group_by": "Region",
                      "column": "Sales", "aggregation": "sum", "sort": "desc", "limit": 2})
    assert r["rows"] == [["South", 400.0], ["East", 350.0]]


def test_groupby_count_without_column(df):
    r = run_plan(df, {"operation": "groupby_aggregate", "group_by": "Category",
                      "aggregation": "count", "sort": "desc"})
    assert r["rows"][0] == ["Tech", 4]


def test_time_trend_month(df):
    r = run_plan(df, {"operation": "time_trend", "date_column": "Order Date",
                      "column": "Sales", "aggregation": "sum", "freq": "month"})
    assert r["rows"][0] == ["2016-01", 300.0]
    assert len(r["rows"]) == 4


def test_top_n_rows(df):
    r = run_plan(df, {"operation": "top_n", "sort_by": "Profit", "n": 2,
                      "columns": ["Region", "Profit"]})
    assert r["rows"] == [["South", 40.0], ["East", 30.0]]


def test_missing_values(df):
    assert run_plan(df, {"operation": "missing_values", "column": "Sales"})["value"] == 1
    table = run_plan(df, {"operation": "missing_values"})
    assert ["Sales", 1] in table["rows"]


def test_unknown_column_rejected(df):
    with pytest.raises(EngineError):
        run_plan(df, {"operation": "aggregate", "column": "Salary", "aggregation": "sum"})


def test_sum_of_text_rejected(df):
    with pytest.raises(EngineError):
        run_plan(df, {"operation": "aggregate", "column": "Region", "aggregation": "sum"})


def test_unknown_operation_rejected(df):
    with pytest.raises(EngineError):
        run_plan(df, {"operation": "drop_table"})


def test_empty_filter_result(df):
    with pytest.raises(EngineError):
        run_plan(df, {"operation": "count_rows",
                      "filters": [{"column": "Region", "op": "==", "value": "Mars"}]})
