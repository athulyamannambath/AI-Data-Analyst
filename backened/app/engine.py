"""Analysis engine: runs a structured JSON plan on a DataFrame with Pandas.

The LLM never writes code. It only produces a plan like:
    {"operation": "groupby_aggregate", "group_by": "Region",
     "column": "Sales", "aggregation": "sum", "sort": "desc", "limit": 5}
and this module executes it using fixed, known operations.

Every plan may also carry "filters": [{"column", "op", "value"}].
"""
import math

import pandas as pd

OPERATIONS = {
    "count_rows", "missing_values", "aggregate",
    "groupby_aggregate", "top_n", "time_trend",
}
AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count", "nunique"}
NUMERIC_ONLY_AGGS = {"sum", "mean", "median"}
FILTER_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "contains"}
FREQ_CODES = {"day": "D", "month": "M", "quarter": "Q", "year": "Y"}
MAX_ROWS = 100


class EngineError(Exception):
    """A plan could not be executed. The message is safe to show the user."""


# ---------- helpers ----------

def _clean(v):
    """Make a value JSON-safe: numpy -> python, NaN -> None, dates -> ISO text."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if hasattr(v, "item"):  # numpy scalar
        v = v.item()
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 2)
    return v


def _require_column(df: pd.DataFrame, name) -> str:
    if name not in df.columns:
        raise EngineError(f"Column '{name}' does not exist in this dataset.")
    return name


def _is_text(series: pd.Series) -> bool:
    return not (pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_bool_dtype(series))


def _check_agg(df: pd.DataFrame, column, aggregation) -> None:
    if aggregation not in AGGREGATIONS:
        raise EngineError(f"Unsupported aggregation '{aggregation}'.")
    if aggregation in NUMERIC_ONLY_AGGS and not pd.api.types.is_numeric_dtype(df[column]):
        raise EngineError(f"Cannot compute {aggregation} of '{column}' because it is not numeric.")


def _table(columns, rows) -> dict:
    total = len(rows)
    return {
        "kind": "table",
        "columns": list(columns),
        "rows": [[_clean(v) for v in r] for r in rows[:MAX_ROWS]],
        "row_count": total,
        "truncated": total > MAX_ROWS,
    }


def _scalar(label, value) -> dict:
    return {"kind": "scalar", "label": label, "value": _clean(value)}


def apply_filters(df: pd.DataFrame, filters) -> pd.DataFrame:
    for f in filters or []:
        col = _require_column(df, f.get("column"))
        op, val = f.get("op"), f.get("value")
        if op not in FILTER_OPS:
            raise EngineError(f"Unsupported filter operator '{op}'.")
        s = df[col]
        try:
            if pd.api.types.is_datetime64_any_dtype(s):
                conv = (lambda x: pd.to_datetime(x)) 
                val = [conv(x) for x in val] if op == "in" else conv(val)
            elif pd.api.types.is_numeric_dtype(s) and op != "contains":
                val = [float(x) for x in val] if op == "in" else float(val)
        except (ValueError, TypeError):
            raise EngineError(f"Filter value {val!r} does not fit column '{col}'.")

        if op == "contains":
            mask = s.astype("string").str.contains(str(val), case=False, regex=False, na=False)
        elif op == "in":
            if _is_text(s):
                wanted = {str(x).strip().lower() for x in val}
                mask = s.astype("string").str.strip().str.lower().isin(wanted)
            else:
                mask = s.isin(val)
        elif op in ("==", "!=") and _is_text(s):
            same = s.astype("string").str.strip().str.lower() == str(val).strip().lower()
            mask = same if op == "==" else ~same
        else:
            mask = {"==": s == val, "!=": s != val, ">": s > val,
                    ">=": s >= val, "<": s < val, "<=": s <= val}[op]
        df = df[mask.fillna(False)]
    return df


# ---------- operations ----------

def _count_rows(df, plan):
    return _scalar("Number of rows", len(df))


def _missing_values(df, plan):
    col = plan.get("column")
    if col:
        _require_column(df, col)
        return _scalar(f"Missing values in {col}", int(df[col].isna().sum()))
    counts = df.isna().sum()
    return _table(["Column", "Missing values"], [[c, int(n)] for c, n in counts.items()])


def _aggregate(df, plan):
    col = _require_column(df, plan.get("column"))
    agg = plan.get("aggregation")
    _check_agg(df, col, agg)
    return _scalar(f"{agg} of {col}", getattr(df[col], agg)())


def _groupby_aggregate(df, plan):
    keys = plan.get("group_by")
    keys = [keys] if isinstance(keys, str) else list(keys or [])
    if not keys:
        raise EngineError("A group_by column is required.")
    for k in keys:
        _require_column(df, k)

    agg, col = plan.get("aggregation"), plan.get("column")
    if col is None and agg == "count":
        out = df.groupby(keys, dropna=True).size().reset_index(name="count")
        value_name = "count"
    else:
        _require_column(df, col)
        _check_agg(df, col, agg)
        value_name = f"{col} ({agg})"
        out = df.groupby(keys, dropna=True)[col].agg(agg).reset_index(name=value_name)

    if plan.get("sort") in ("asc", "desc"):
        out = out.sort_values(value_name, ascending=plan["sort"] == "asc")
    if plan.get("limit"):
        out = out.head(int(plan["limit"]))
    return _table(out.columns, out.values.tolist())


def _top_n(df, plan):
    sort_by = _require_column(df, plan.get("sort_by"))
    n = int(plan.get("n") or 5)
    ascending = plan.get("order") == "asc"
    cols = plan.get("columns") or list(df.columns)
    for c in cols:
        _require_column(df, c)
    out = df.sort_values(sort_by, ascending=ascending).head(n)[cols]
    return _table(cols, out.values.tolist())


def _time_trend(df, plan):
    dcol = _require_column(df, plan.get("date_column"))
    if not pd.api.types.is_datetime64_any_dtype(df[dcol]):
        raise EngineError(f"'{dcol}' is not a date column.")
    freq = plan.get("freq", "month")
    if freq not in FREQ_CODES:
        raise EngineError(f"Unsupported frequency '{freq}'.")
    agg, col = plan.get("aggregation", "sum"), plan.get("column")

    d = df.dropna(subset=[dcol]).copy()
    d["_period"] = d[dcol].dt.to_period(FREQ_CODES[freq])
    if col is None and agg == "count":
        out = d.groupby("_period").size().reset_index(name="count")
        value_name = "count"
    else:
        _require_column(d, col)
        _check_agg(d, col, agg)
        value_name = f"{col} ({agg})"
        out = d.groupby("_period")[col].agg(agg).reset_index(name=value_name)
    out = out.sort_values("_period")
    out["_period"] = out["_period"].astype(str)
    return _table([freq.capitalize(), value_name], out.values.tolist())


_HANDLERS = {
    "count_rows": _count_rows,
    "missing_values": _missing_values,
    "aggregate": _aggregate,
    "groupby_aggregate": _groupby_aggregate,
    "top_n": _top_n,
    "time_trend": _time_trend,
}


def run_plan(df: pd.DataFrame, plan: dict) -> dict:
    """Execute one plan. Returns {"kind": "scalar"|"table", ...} or raises EngineError."""
    op = plan.get("operation")
    if op not in OPERATIONS:
        raise EngineError(f"Unsupported operation '{op}'.")
    filtered = apply_filters(df, plan.get("filters"))
    if op != "missing_values" and len(filtered) == 0:
        raise EngineError("No rows match the filters in this question.")
    return _HANDLERS[op](filtered, plan)
