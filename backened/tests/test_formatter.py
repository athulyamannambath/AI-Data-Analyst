from app.formatter import format_response


def table(columns, rows, **kw):
    return {"kind": "table", "columns": columns, "rows": rows,
            "row_count": kw.get("row_count", len(rows)), "truncated": kw.get("truncated", False)}


GROUP_PLAN = {"operation": "groupby_aggregate", "group_by": ["Region"], "column": "Sales",
              "aggregation": "sum", "sort": "desc", "filters": []}


def test_scalar_aggregate():
    r = format_response({"kind": "scalar", "label": "mean of Sales", "value": 1234.5},
                        {"operation": "aggregate", "column": "Sales", "aggregation": "mean", "filters": []})
    assert r["result_type"] == "number" and r["value"] == 1234.5
    assert r["explanation"] == "Average Sales is 1,234.50."


def test_scalar_count_with_filter():
    plan = {"operation": "count_rows",
            "filters": [{"column": "Region", "op": "==", "value": "West"}]}
    r = format_response({"kind": "scalar", "label": "Number of rows", "value": 3140}, plan)
    assert r["explanation"] == "There are 3,140 rows. Filter: Region == West."


def test_chart_for_grouped_table():
    plan = {**GROUP_PLAN, "output": "chart", "chart_type": "bar"}
    r = format_response(table(["Region", "Sales (sum)"], [["West", 725457.82], ["East", 678781.24]]), plan)
    assert r["result_type"] == "chart"
    assert r["chart"] == {"type": "bar", "x": "Region", "y": "Sales (sum)"}
    assert "West comes first at 725,457.82" in r["explanation"]


def test_table_when_no_chart_requested():
    r = format_response(table(["Region", "Sales (sum)"], [["West", 1.0]]), GROUP_PLAN)
    assert r["result_type"] == "table" and r["chart"] is None


def test_time_trend_defaults_to_line_chart():
    plan = {"operation": "time_trend", "column": "Sales", "aggregation": "sum",
            "freq": "month", "filters": []}
    r = format_response(table(["Month", "Sales (sum)"], [["2016-01", 10.0], ["2016-02", 30.0]]), plan)
    assert r["chart"]["type"] == "line"
    assert "from 2016-01 to 2016-02" in r["explanation"]
    assert "highest was 2016-02 at 30.00" in r["explanation"]


def test_two_group_columns_fall_back_to_table_with_note():
    plan = {**GROUP_PLAN, "group_by": ["Region", "Category"], "output": "chart"}
    r = format_response(table(["Region", "Category", "Sales (sum)"], [["West", "Tech", 1.0]]), plan)
    assert r["result_type"] == "table" and r["chart"] is None
    assert "shown as a table" in r["explanation"]


def test_pie_with_too_many_slices_becomes_bar():
    rows = [[f"c{i}", float(i)] for i in range(12)]
    plan = {**GROUP_PLAN, "output": "chart", "chart_type": "pie"}
    assert format_response(table(["A", "B"], rows), plan)["chart"]["type"] == "bar"


def test_truncation_note():
    plan = {"operation": "top_n", "sort_by": "Profit", "n": 100, "order": "desc", "filters": []}
    r = format_response(table(["Profit"], [[1.0]] * 100, row_count=250, truncated=True), plan)
    assert "Showing the first 100 of 250 rows" in r["explanation"]
