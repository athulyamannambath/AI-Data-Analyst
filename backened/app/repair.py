"""Plan repair: fixes known, unambiguous small-model mistakes on a plan that has already
passed the validator. Repaired plans are built only from validated columns and allowed
operations, so they need no second validation. The plan shown in every answer is the
repaired one, so nothing is hidden.

Repairs
1. top_n listing a single category column  ->  groupby_aggregate
     {"operation": "top_n", "sort_by": "Sales", "n": 5, "columns": ["State"]}
   means "rank the states by sales", not "list the state names of 5 single orders".
   Uses mean when the question says average/mean, otherwise sum.

2. groupby_aggregate where the question names ONE value of the grouping column  ->  filter
     "average math score for students who completed the course", grouped by course
   means "only the completed group", so it becomes aggregate/count_rows with that filter.
   Skipped when the question asks for a breakdown ("by", "each", "per", "compare", ...),
   when it mentions two or more values, or when the column is already filtered.

3. groupby_aggregate over a column that is already filtered to ONE value  ->  single number
     filters: gender == female, group_by: gender
   can only ever produce one group, so the answer is one number, not a one-row table.
"""
import re

import pandas as pd

_AVERAGE = re.compile(r"\b(average|averages|mean|avg)\b", re.IGNORECASE)
_BREAKDOWN = re.compile(
    r"\b(by|per|each|every|versus|vs|compare|compared|comparison|breakdown|split|across)\b", re.IGNORECASE)
_GENERIC_LABELS = {"yes", "no", "none", "other", "all", "true", "false"}
MAX_LABELS = 20
MAX_FILTERS = 5
CARRY_OVER = ("output", "chart_type")


def _rank_groups(plan: dict, question: str, df: pd.DataFrame) -> dict:
    if plan.get("operation") != "top_n":
        return plan

    columns, sort_by = plan.get("columns"), plan["sort_by"]
    if not columns or len(columns) != 1 or columns[0] == sort_by:
        return plan
    group = columns[0]
    if pd.api.types.is_numeric_dtype(df[group]) or pd.api.types.is_datetime64_any_dtype(df[group]):
        return plan                        # a numeric/date column is a real row listing
    if not pd.api.types.is_numeric_dtype(df[sort_by]):
        return plan                        # cannot sum or average dates

    fixed = {
        "operation": "groupby_aggregate",
        "filters": plan["filters"],
        "group_by": [group],
        "column": sort_by,
        "aggregation": "mean" if _AVERAGE.search(question or "") else "sum",
        "sort": plan["order"],
        "limit": plan["n"],
    }
    fixed.update({k: plan[k] for k in CARRY_OVER if k in plan})
    return fixed


def _mentioned_labels(df: pd.DataFrame, column: str, question: str) -> list[str]:
    s = df[column]
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return []
    values = s.dropna().astype(str).unique()
    if len(values) > MAX_LABELS:
        return []
    return [v for v in values
            if len(v) >= 3 and v.lower() not in _GENERIC_LABELS
            and re.search(rf"(?<!\w){re.escape(v)}(?!\w)", question, re.IGNORECASE)]


def _single_group_to_filter(plan: dict, question: str, df: pd.DataFrame) -> dict:
    if plan.get("operation") != "groupby_aggregate" or len(plan["group_by"]) != 1:
        return plan
    group = plan["group_by"][0]
    if any(f["column"] == group for f in plan["filters"]):
        return plan
    if not question or _BREAKDOWN.search(question):
        return plan
    labels = _mentioned_labels(df, group, question)
    if len(labels) != 1:
        return plan
    filters = plan["filters"] + [{"column": group, "op": "==", "value": labels[0]}]
    if len(filters) > MAX_FILTERS:
        return plan
    if plan.get("column") is None:                       # a count of rows
        return {"operation": "count_rows", "filters": filters}
    return {"operation": "aggregate", "filters": filters,
            "column": plan["column"], "aggregation": plan["aggregation"]}


def _collapse_filtered_group(plan: dict) -> dict:
    if plan.get("operation") != "groupby_aggregate" or len(plan["group_by"]) != 1:
        return plan
    group = plan["group_by"][0]
    one_value = any(
        f["column"] == group and (f["op"] == "==" or (f["op"] == "in" and len(f["value"]) == 1))
        for f in plan["filters"]
    )
    if not one_value:
        return plan
    if plan.get("column") is None:                       # a count of rows
        return {"operation": "count_rows", "filters": plan["filters"]}
    return {"operation": "aggregate", "filters": plan["filters"],
            "column": plan["column"], "aggregation": plan["aggregation"]}


def repair_plan(plan: dict, question: str, df: pd.DataFrame) -> dict:
    plan = _rank_groups(plan, question, df)
    plan = _single_group_to_filter(plan, question, df)
    return _collapse_filtered_group(plan)