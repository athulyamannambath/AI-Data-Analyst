"""Plan validator: the safety gate between the LLM and the engine.

validate_plan(plan, df) returns a cleaned copy of the plan with exact column
names, or raises ValidationError with a message that is safe to show the user.
Anything not explicitly allowed here is rejected.
"""
import re

import pandas as pd

from app.engine import AGGREGATIONS, FILTER_OPS, FREQ_CODES, NUMERIC_ONLY_AGGS, OPERATIONS

OUTPUTS = {"number", "table", "chart"}
CHART_TYPES = {"bar", "line", "pie"}
MAX_FILTERS = 5
MAX_LIMIT = 100

COMMON_KEYS = {"operation", "filters", "output", "chart_type"}
ALLOWED_KEYS = {
    "count_rows": set(),
    "missing_values": {"column"},
    "aggregate": {"column", "aggregation"},
    "groupby_aggregate": {"group_by", "column", "aggregation", "sort", "limit"},
    "top_n": {"sort_by", "n", "order", "columns"},
    "time_trend": {"date_column", "column", "aggregation", "freq"},
}
SCALAR_OPS = {"count_rows", "aggregate"}


class ValidationError(Exception):
    """The plan is not allowed. The message is safe to show the user."""


# ---------- helpers ----------

def _norm(name) -> str:
    """Compare names ignoring case, spaces, underscores, hyphens and other symbols."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve_column(aliases: dict, df: pd.DataFrame, name) -> str:
    """Exact match, then a unique loose match (see _norm), then a dataset alias. Otherwise reject."""
    if not isinstance(name, str) or not name.strip():
        raise ValidationError("A column name is missing or invalid.")
    if name in df.columns:
        return name
    matches = [c for c in df.columns if _norm(c) == _norm(name)]
    if len(matches) == 1:
        return matches[0]
    target = aliases.get(_norm(name))      # synonyms from datasets.json, e.g. salary -> MonthlyIncome
    if target in df.columns:
        return target
    raise ValidationError(
        f"I can't find a column called '{name}' in this dataset. "
        f"Available columns: {', '.join(df.columns)}."
    )


def _is_numeric(s): return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
def _is_date(s): return pd.api.types.is_datetime64_any_dtype(s)


def _positive_int(value, field, maximum=MAX_LIMIT) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValidationError(f"'{field}' must be a whole number.")
    if not 1 <= n <= maximum:
        raise ValidationError(f"'{field}' must be between 1 and {maximum}.")
    return n


def _number(value, column):
    if isinstance(value, bool):
        raise ValidationError(f"Filter value for '{column}' must be a number.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"Filter value {value!r} is not a number, but '{column}' is numeric.")


def _scalar_value(value, column):
    if isinstance(value, (list, dict)) or value is None:
        raise ValidationError(f"Filter value for '{column}' is invalid.")
    return value


# ---------- pieces ----------

def _validate_filters(df, filters, aliases) -> list:
    if filters in (None, []):
        return []
    if not isinstance(filters, list):
        raise ValidationError("Filters must be a list.")
    if len(filters) > MAX_FILTERS:
        raise ValidationError(f"At most {MAX_FILTERS} filters are allowed.")

    clean = []
    for f in filters:
        if not isinstance(f, dict) or set(f) - {"column", "op", "value"}:
            raise ValidationError("Each filter needs exactly: column, op, value.")
        col = _resolve_column(aliases, df, f.get("column"))
        op = f.get("op")
        if op not in FILTER_OPS:
            raise ValidationError(f"Filter operator {op!r} is not supported.")
        value, s = f.get("value"), df[col]

        if _is_numeric(s):
            if op == "contains":
                raise ValidationError(f"'contains' cannot be used on numeric column '{col}'.")
            value = [_number(v, col) for v in value] if op == "in" and isinstance(value, list) \
                else _number(value, col)
        elif _is_date(s):
            if op == "contains":
                raise ValidationError(f"'contains' cannot be used on date column '{col}'.")
            try:
                for v in (value if op == "in" and isinstance(value, list) else [value]):
                    pd.to_datetime(v)
            except (ValueError, TypeError):
                raise ValidationError(f"Filter value {value!r} is not a valid date for '{col}'.")
        else:  # text
            if op in (">", ">=", "<", "<="):
                raise ValidationError(f"'{op}' cannot be used on text column '{col}'.")

        if op == "in":
            if not isinstance(value, list) or not value or len(value) > 20:
                raise ValidationError("'in' needs a list of 1 to 20 values.")
            value = [_scalar_value(v, col) for v in value]
        else:
            value = _scalar_value(value, col)
        clean.append({"column": col, "op": op, "value": value})
    return clean


def _check_agg(df, column, aggregation):
    if aggregation not in AGGREGATIONS:
        raise ValidationError(f"Aggregation {aggregation!r} is not supported.")
    if column is None:
        if aggregation != "count":
            raise ValidationError(f"'{aggregation}' needs a column.")
        return
    if aggregation in NUMERIC_ONLY_AGGS and not _is_numeric(df[column]):
        raise ValidationError(f"Cannot compute {aggregation} of '{column}' because it is not numeric.")
    if aggregation in ("min", "max") and not (_is_numeric(df[column]) or _is_date(df[column])):
        raise ValidationError(f"Cannot compute {aggregation} of '{column}' because it is text.")


# ---------- main entry ----------

def validate_plan(plan, df: pd.DataFrame, aliases: dict | None = None) -> dict:
    aliases = {_norm(k): v for k, v in (aliases or {}).items()}
    if not isinstance(plan, dict):
        raise ValidationError("The analysis plan is not in a valid format.")
    op = plan.get("operation")
    if op not in OPERATIONS:
        raise ValidationError("That type of analysis is not supported.")

    # "Average profit by segment" written as aggregate + group_by is exactly groupby_aggregate.
    if op == "aggregate" and plan.get("group_by") is not None:
        plan = {**plan, "operation": "groupby_aggregate"}
        op = "groupby_aggregate"

    extra = set(plan) - COMMON_KEYS - ALLOWED_KEYS[op]
    if extra:
        raise ValidationError(f"The plan contains unsupported fields: {', '.join(sorted(extra))}.")

    out = {"operation": op, "filters": _validate_filters(df, plan.get("filters"), aliases)}

    if op == "missing_values":
        if plan.get("column") is not None:
            out["column"] = _resolve_column(aliases, df, plan["column"])

    elif op == "aggregate":
        out["column"] = _resolve_column(aliases, df, plan.get("column"))
        out["aggregation"] = plan.get("aggregation")
        _check_agg(df, out["column"], out["aggregation"])

    elif op == "groupby_aggregate":
        keys = plan.get("group_by")
        keys = [keys] if isinstance(keys, str) else keys
        if not isinstance(keys, list) or not 1 <= len(keys) <= 2:
            raise ValidationError("Choose one or two columns to group by.")
        out["group_by"] = [_resolve_column(aliases, df, k) for k in keys]
        out["aggregation"] = plan.get("aggregation")
        out["column"] = _resolve_column(aliases, df, plan["column"]) if plan.get("column") is not None else None
        _check_agg(df, out["column"], out["aggregation"])
        if plan.get("sort") is not None:
            if plan["sort"] not in ("asc", "desc"):
                raise ValidationError("'sort' must be 'asc' or 'desc'.")
            out["sort"] = plan["sort"]
        if plan.get("limit") is not None:
            out["limit"] = _positive_int(plan["limit"], "limit")

    elif op == "top_n":
        out["sort_by"] = _resolve_column(aliases, df, plan.get("sort_by"))
        if not (_is_numeric(df[out["sort_by"]]) or _is_date(df[out["sort_by"]])):
            raise ValidationError(f"Cannot rank rows by text column '{out['sort_by']}'.")
        out["n"] = _positive_int(plan.get("n", 5), "n")
        if plan.get("order") not in (None, "asc", "desc"):
            raise ValidationError("'order' must be 'asc' or 'desc'.")
        out["order"] = plan.get("order") or "desc"
        if plan.get("columns") is not None:
            if not isinstance(plan["columns"], list) or not plan["columns"]:
                raise ValidationError("'columns' must be a non-empty list.")
            out["columns"] = [_resolve_column(aliases, df, c) for c in plan["columns"]]
        # "Highest X" asked as top_n(n=1) of one column is exactly max/min of that column.
        # Small models do this often, so return the single number that was meant.
        if out["n"] == 1 and out.get("columns") == [out["sort_by"]]:
            out = {"operation": "aggregate", "filters": out["filters"], "column": out["sort_by"],
                   "aggregation": "max" if out["order"] == "desc" else "min"}
            op = "aggregate"

    elif op == "time_trend":
        out["date_column"] = _resolve_column(aliases, df, plan.get("date_column"))
        if not _is_date(df[out["date_column"]]):
            raise ValidationError(f"'{out['date_column']}' is not a date column, so I can't show a trend over time.")
        out["freq"] = plan.get("freq", "month")
        if out["freq"] not in FREQ_CODES:
            raise ValidationError("Trend frequency must be day, month, quarter or year.")
        out["aggregation"] = plan.get("aggregation", "sum")
        out["column"] = _resolve_column(aliases, df, plan["column"]) if plan.get("column") is not None else None
        _check_agg(df, out["column"], out["aggregation"])

    # presentation hints
    output = plan.get("output")
    if output is not None and output not in OUTPUTS:
        raise ValidationError("'output' must be number, table or chart.")
    chart_type = plan.get("chart_type")
    if chart_type is not None and chart_type not in CHART_TYPES:
        raise ValidationError("'chart_type' must be bar, line or pie.")
    if output == "chart" and op in SCALAR_OPS:
        raise ValidationError("A single number can't be shown as a chart.")
    if output:
        out["output"] = output
    if chart_type:
        out["chart_type"] = chart_type
    return out