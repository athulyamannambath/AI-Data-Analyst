import pandas as pd

from app.repair import repair_plan
from app.validator import validate_plan

DF = pd.DataFrame({
    "State": ["CA", "CA", "NY", "TX"],
    "Sales": [100.0, 200.0, 250.0, 50.0],
    "Qty": [1, 2, 3, 4],
    "Order Date": pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01", "2016-04-01"]),
})


def fix(question="top 3 states by total sales", **plan):
    base = {"operation": "top_n", "sort_by": "Sales", "n": 3, "order": "desc", "filters": []}
    base.update(plan)
    return repair_plan(base, question, DF)


def test_single_category_column_becomes_group_ranking():
    p = fix(columns=["State"])
    assert p == {"operation": "groupby_aggregate", "filters": [], "group_by": ["State"],
                 "column": "Sales", "aggregation": "sum", "sort": "desc", "limit": 3}


def test_average_wording_uses_mean():
    assert fix("top 3 states by average sales", columns=["State"])["aggregation"] == "mean"
    assert fix("Which states have the highest mean sales?", columns=["State"])["aggregation"] == "mean"


def test_filters_order_and_chart_hints_are_kept():
    flt = [{"column": "Qty", "op": ">", "value": 1}]
    p = fix(columns=["State"], filters=flt, order="asc", output="chart", chart_type="bar")
    assert p["filters"] == flt and p["sort"] == "asc" and p["output"] == "chart" and p["chart_type"] == "bar"


def test_real_row_listings_are_left_alone():
    assert fix(columns=["State", "Sales"])["operation"] == "top_n"      # two columns
    assert fix()["operation"] == "top_n"                                  # no columns
    assert fix(columns=["Sales"])["operation"] == "top_n"                 # same as sort_by
    assert fix(columns=["Qty"])["operation"] == "top_n"                   # numeric column
    assert fix(columns=["Order Date"])["operation"] == "top_n"            # date column


def test_date_sort_is_left_alone():
    assert fix(sort_by="Order Date", columns=["State"])["operation"] == "top_n"


def test_other_operations_are_untouched():
    plan = {"operation": "aggregate", "column": "Sales", "aggregation": "sum", "filters": []}
    assert repair_plan(plan, "total sales", DF) is plan


def test_repair_composes_with_validator_end_to_end():
    raw = {"operation": "top_n", "sort_by": "sales", "n": 2, "columns": ["state"]}
    clean = repair_plan(validate_plan(raw, DF), "top 2 states by total sales", DF)
    assert clean["operation"] == "groupby_aggregate" and clean["group_by"] == ["State"]


# ---------- repair 2: one named category value -> filter ----------

STU = pd.DataFrame({
    "course": ["completed", "none", "completed", "none"],
    "gender": ["female", "male", "female", "male"],
    "math": [70.0, 60.0, 90.0, 50.0],
    "state": [f"state{i}" for i in range(4)],
})


def group_plan(**kw):
    base = {"operation": "groupby_aggregate", "filters": [], "group_by": ["course"],
            "column": "math", "aggregation": "mean"}
    base.update(kw)
    return base


def regroup(question, df=STU, **kw):
    return repair_plan(group_plan(**kw), question, df)


def test_named_value_becomes_filter_and_scalar():
    p = regroup("What is the average math score for students who completed the test preparation course?")
    assert p == {"operation": "aggregate", "column": "math", "aggregation": "mean",
                 "filters": [{"column": "course", "op": "==", "value": "completed"}]}


def test_match_is_case_insensitive_and_keeps_existing_filters():
    flt = [{"column": "gender", "op": "==", "value": "female"}]
    p = regroup("average math for COMPLETED students", filters=flt)
    assert p["filters"] == flt + [{"column": "course", "op": "==", "value": "completed"}]


def test_count_without_column_becomes_count_rows():
    p = regroup("how many students completed the course", column=None, aggregation="count")
    assert p == {"operation": "count_rows",
                 "filters": [{"column": "course", "op": "==", "value": "completed"}]}


def test_chart_hints_are_dropped_for_single_value():
    p = regroup("average math for completed students", output="chart", chart_type="bar")
    assert "output" not in p and "chart_type" not in p


def test_breakdown_words_block_the_repair():
    for q in ("average math by course, completed first", "average math for each course (completed or not)",
              "compare completed with the rest", "average math per course for completed students"):
        assert regroup(q)["operation"] == "groupby_aggregate", q


def test_no_label_or_two_labels_block_the_repair():
    assert regroup("average math score by gender")["operation"] == "groupby_aggregate"
    three = pd.DataFrame({"course": ["completed", "started", "pending", "completed"], "math": [1.0, 2.0, 3.0, 4.0]})
    assert regroup("average math for completed and started students", df=three)["operation"] == "groupby_aggregate"
    # "none" is too generic to count as a label, so this names only one value
    assert regroup("average math for completed and none")["operation"] == "aggregate"


def test_existing_filter_on_group_column_is_not_added_twice():
    flt = [{"column": "course", "op": "==", "value": "completed"}]
    p = regroup("average math for completed students", filters=flt)
    assert p["operation"] == "aggregate" and p["filters"] == flt      # collapsed by repair 3, no duplicate filter


def test_generic_labels_and_high_cardinality_columns_are_ignored():
    yn = pd.DataFrame({"flag": ["Yes", "No", "Yes"], "math": [1.0, 2.0, 3.0]})
    p = repair_plan(group_plan(group_by=["flag"]), "average math for Yes answers", yn)
    assert p["operation"] == "groupby_aggregate"
    many = pd.DataFrame({"k": [f"label{i}" for i in range(25)], "math": [1.0] * 25})
    p = repair_plan(group_plan(group_by=["k"]), "average math for label3", many)
    assert p["operation"] == "groupby_aggregate"


def test_two_group_columns_are_left_alone():
    assert regroup("average math for completed", group_by=["course", "gender"])["operation"] == "groupby_aggregate"


# ---------- repair 3: group column already filtered to one value -> single number ----------

def test_group_filtered_to_one_value_becomes_single_number():
    # the exact plan the model produced for "female students who scored above 80 in reading"
    plan = {"operation": "groupby_aggregate", "group_by": ["gender"], "column": "math", "aggregation": "count",
            "filters": [{"column": "gender", "op": "==", "value": "female"},
                        {"column": "math", "op": ">", "value": 80.0}]}
    p = repair_plan(plan, "How many female students scored above 80 in reading?", STU)
    assert p == {"operation": "aggregate", "column": "math", "aggregation": "count", "filters": plan["filters"]}


def test_collapse_count_without_column_and_drops_hints():
    plan = {"operation": "groupby_aggregate", "group_by": ["gender"], "column": None, "aggregation": "count",
            "filters": [{"column": "gender", "op": "==", "value": "female"}],
            "sort": "desc", "limit": 3, "output": "chart", "chart_type": "bar"}
    assert repair_plan(plan, "q", STU) == {"operation": "count_rows", "filters": plan["filters"]}


def test_collapse_works_for_in_with_one_value_only():
    base = {"operation": "groupby_aggregate", "group_by": ["gender"], "column": "math", "aggregation": "mean"}
    one = repair_plan({**base, "filters": [{"column": "gender", "op": "in", "value": ["female"]}]}, "q", STU)
    two = repair_plan({**base, "filters": [{"column": "gender", "op": "in", "value": ["female", "male"]}]}, "q", STU)
    assert one["operation"] == "aggregate" and two["operation"] == "groupby_aggregate"


def test_collapse_does_not_touch_real_groupings():
    base = {"operation": "groupby_aggregate", "group_by": ["gender"], "column": "math", "aggregation": "mean"}
    other = repair_plan({**base, "filters": [{"column": "course", "op": "==", "value": "completed"}]}, "q", STU)
    neq = repair_plan({**base, "filters": [{"column": "gender", "op": "!=", "value": "male"}]}, "q", STU)
    two_keys = repair_plan({**base, "group_by": ["gender", "course"],
                            "filters": [{"column": "gender", "op": "==", "value": "female"}]}, "q", STU)
    assert other["operation"] == neq["operation"] == two_keys["operation"] == "groupby_aggregate"