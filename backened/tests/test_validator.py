import pandas as pd
import pytest

from app.validator import ValidationError, validate_plan


@pytest.fixture
def df():
    return pd.DataFrame({
        "Region": ["West", "East"],
        "math score": [70, 80],
        "Sales": [100.0, 200.0],
        "Order Date": pd.to_datetime(["2016-01-05", "2016-02-10"]),
    })


def ok(df, **plan):
    return validate_plan(plan, df)


def bad(df, **plan):
    with pytest.raises(ValidationError):
        validate_plan(plan, df)


def test_valid_plan_passes(df):
    p = ok(df, operation="groupby_aggregate", group_by="Region", column="Sales",
           aggregation="sum", sort="desc", limit=3)
    assert p["group_by"] == ["Region"] and p["limit"] == 3


def test_column_names_are_corrected(df):
    p = ok(df, operation="aggregate", column="MATH  score", aggregation="mean")
    assert p["column"] == "math score"


def test_unknown_column_rejected(df):
    bad(df, operation="aggregate", column="Salary", aggregation="sum")


def test_unknown_operation_rejected(df):
    bad(df, operation="delete_rows")


def test_non_dict_rejected(df):
    with pytest.raises(ValidationError):
        validate_plan("DROP TABLE", df)


def test_extra_keys_rejected(df):
    bad(df, operation="count_rows", code="__import__('os')")


def test_sum_of_text_rejected(df):
    bad(df, operation="aggregate", column="Region", aggregation="sum")


def test_bad_aggregation_rejected(df):
    bad(df, operation="aggregate", column="Sales", aggregation="exec")


def test_count_without_column_ok_but_sum_without_column_not(df):
    ok(df, operation="groupby_aggregate", group_by="Region", aggregation="count")
    bad(df, operation="groupby_aggregate", group_by="Region", aggregation="sum")


def test_limit_bounds(df):
    bad(df, operation="groupby_aggregate", group_by="Region", column="Sales",
        aggregation="sum", limit=100000)
    bad(df, operation="groupby_aggregate", group_by="Region", column="Sales",
        aggregation="sum", limit="abc")


def test_filter_checks(df):
    ok(df, operation="count_rows", filters=[{"column": "region", "op": "==", "value": "West"}])
    bad(df, operation="count_rows", filters=[{"column": "Sales", "op": ">", "value": "lots"}])
    bad(df, operation="count_rows", filters=[{"column": "Region", "op": ">", "value": "A"}])
    bad(df, operation="count_rows", filters=[{"column": "Region", "op": "like", "value": "W"}])
    bad(df, operation="count_rows", filters=[{"column": "Region", "op": "==", "value": "W", "x": 1}])
    bad(df, operation="count_rows", filters=[{"column": "Region", "op": "==", "value": "W"}] * 6)


def test_time_trend_needs_date_column(df):
    ok(df, operation="time_trend", date_column="Order Date", column="Sales", aggregation="sum")
    bad(df, operation="time_trend", date_column="Region", column="Sales", aggregation="sum")
    bad(df, operation="time_trend", date_column="Order Date", column="Sales",
        aggregation="sum", freq="decade")


def test_top_n_rejects_text_sort(df):
    bad(df, operation="top_n", sort_by="Region")


def test_output_and_chart_hints(df):
    ok(df, operation="groupby_aggregate", group_by="Region", column="Sales",
       aggregation="sum", output="chart", chart_type="bar")
    bad(df, operation="aggregate", column="Sales", aggregation="sum", output="chart")
    bad(df, operation="groupby_aggregate", group_by="Region", column="Sales",
        aggregation="sum", chart_type="3d")


def test_loose_column_matching(df):
    # underscores, spaces, case and hyphens are all ignored
    assert ok(df, operation="aggregate", column="math_score", aggregation="mean")["column"] == "math score"
    assert ok(df, operation="aggregate", column="Math-Score", aggregation="mean")["column"] == "math score"
    assert ok(df, operation="time_trend", date_column="order_date", column="sales",
              aggregation="sum")["date_column"] == "Order Date"


def test_loose_match_works_for_camelcase_columns():
    hr = pd.DataFrame({"MonthlyIncome": [1, 2], "JobRole": ["a", "b"]})
    p = validate_plan({"operation": "groupby_aggregate", "group_by": "job role",
                       "column": "monthly income", "aggregation": "mean"}, hr)
    assert p["group_by"] == ["JobRole"] and p["column"] == "MonthlyIncome"


def test_ambiguous_loose_match_is_rejected():
    amb = pd.DataFrame({"x y": [1], "x_y": [2]})
    # exact name still works
    assert validate_plan({"operation": "aggregate", "column": "x_y", "aggregation": "sum"}, amb)["column"] == "x_y"
    # a name that loosely matches both is refused rather than guessed
    with pytest.raises(ValidationError):
        validate_plan({"operation": "aggregate", "column": "X-Y", "aggregation": "sum"}, amb)


def test_top_one_of_single_column_becomes_max_or_min(df):
    p = ok(df, operation="top_n", sort_by="math score", n=1, order="desc", columns=["math score"])
    assert p["operation"] == "aggregate" and p["aggregation"] == "max" and p["column"] == "math score"
    p = ok(df, operation="top_n", sort_by="Sales", n=1, order="asc", columns=["Sales"],
           filters=[{"column": "Region", "op": "==", "value": "West"}])
    assert p["aggregation"] == "min" and len(p["filters"]) == 1


def test_top_n_with_real_rows_is_left_alone(df):
    assert ok(df, operation="top_n", sort_by="Sales", n=1, columns=["Region", "Sales"])["operation"] == "top_n"
    assert ok(df, operation="top_n", sort_by="Sales", n=3, columns=["Sales"])["operation"] == "top_n"
    assert ok(df, operation="top_n", sort_by="Sales", n=1)["operation"] == "top_n"


HR = pd.DataFrame({"MonthlyIncome": [1, 2], "Department": ["a", "b"], "Salary_Grade": ["x", "y"]})


def test_alias_resolves_synonym():
    p = validate_plan({"operation": "groupby_aggregate", "group_by": "Department",
                       "column": "Salary", "aggregation": "mean"}, HR, {"salary": "MonthlyIncome"})
    assert p["column"] == "MonthlyIncome"


def test_alias_works_inside_filters():
    p = validate_plan({"operation": "count_rows",
                       "filters": [{"column": "Pay", "op": ">", "value": 1}]}, HR, {"pay": "MonthlyIncome"})
    assert p["filters"][0]["column"] == "MonthlyIncome"


def test_alias_to_missing_column_is_rejected():
    with pytest.raises(ValidationError):
        validate_plan({"operation": "aggregate", "column": "salary", "aggregation": "sum"},
                      HR, {"salary": "DoesNotExist"})


def test_real_column_beats_alias():
    p = validate_plan({"operation": "aggregate", "column": "Department", "aggregation": "count"},
                      HR, {"department": "MonthlyIncome"})
    assert p["column"] == "Department"


def test_without_aliases_salary_is_still_rejected():
    with pytest.raises(ValidationError):
        validate_plan({"operation": "aggregate", "column": "Salary", "aggregation": "sum"}, HR)


def test_aggregate_with_group_by_becomes_groupby_aggregate(df):
    p = ok(df, operation="aggregate", group_by="Region", column="Sales", aggregation="sum")
    assert p["operation"] == "groupby_aggregate" and p["group_by"] == ["Region"]
    assert p["column"] == "Sales" and p["aggregation"] == "sum"


def test_plain_aggregate_is_unchanged(df):
    assert ok(df, operation="aggregate", column="Sales", aggregation="sum")["operation"] == "aggregate"


def test_aggregate_with_bad_group_by_column_still_rejected(df):
    bad(df, operation="aggregate", group_by="Nope", column="Sales", aggregation="sum")