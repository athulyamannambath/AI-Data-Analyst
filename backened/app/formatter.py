"""Formatter: turns an engine result into the response fields the frontend shows.

Explanations are built from templates and the computed values. The LLM never
writes numbers (FR-11), so the text can't contradict the table.
"""

METRIC = {
    "sum": "Total {c}", "mean": "Average {c}", "median": "Median {c}",
    "min": "Minimum {c}", "max": "Maximum {c}",
    "count": "Count of {c}", "nunique": "Number of distinct {c}",
}
MAX_PIE_SLICES = 8


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def _metric(plan) -> str:
    col, agg = plan.get("column"), plan.get("aggregation")
    if col is None:
        return "Number of rows"
    return METRIC.get(agg, "{c}").format(c=col)


def _filter_text(plan) -> str:
    parts = []
    for f in plan.get("filters") or []:
        v = f["value"]
        v = ", ".join(map(str, v)) if isinstance(v, list) else v
        parts.append(f'{f["column"]} {f["op"]} {v}')
    return f" Filter: {'; '.join(parts)}." if parts else ""


def _chart_spec(plan, result):
    """Return a chart spec, or None. Second item is a note when a chart was wanted but impossible."""
    op = plan["operation"]
    wanted = plan.get("output") == "chart" or (op == "time_trend" and plan.get("output") is None)
    if not wanted or result["kind"] != "table":
        return None, ""
    cols, rows = result["columns"], result["rows"]
    if len(cols) != 2 or not rows:
        return None, " (Charts need one grouping column, so this is shown as a table.)"
    ctype = plan.get("chart_type") or ("line" if op == "time_trend" else "bar")
    if ctype == "pie":
        values = [r[1] for r in rows if isinstance(r[1], (int, float))]
        if len(rows) > MAX_PIE_SLICES or any(v < 0 for v in values):
            ctype = "bar"
    return {"type": ctype, "x": cols[0], "y": cols[1]}, ""


def _explain(plan, result) -> str:
    op, kind = plan["operation"], result["kind"]
    filt = _filter_text(plan)

    if kind == "scalar":
        v = fmt(result["value"])
        if op == "count_rows":
            return f"There are {v} rows." + filt
        if op == "missing_values":
            return f"{plan['column']} has {v} missing values." + filt
        return f"{_metric(plan)} is {v}." + filt

    rows = result["rows"]
    text = ""
    if op == "missing_values":
        text = "Missing values per column."
    elif op == "top_n":
        word = "Bottom" if plan.get("order") == "asc" else "Top"
        text = f"{word} {plan['n']} rows by {plan['sort_by']}."
    elif op == "groupby_aggregate":
        text = f"{_metric(plan)} by {', '.join(plan['group_by'])}"
        if plan.get("sort"):
            text += ", sorted " + ("high to low" if plan["sort"] == "desc" else "low to high")
        if plan.get("limit"):
            text += f", showing the first {plan['limit']}"
        text += "."
        if plan.get("sort") and rows:
            first = rows[0]
            text += f" {', '.join(map(str, first[:-1]))} comes first at {fmt(first[-1])}."
    elif op == "time_trend":
        text = f"{_metric(plan)} per {plan['freq']}"
        if rows:
            text += f", from {rows[0][0]} to {rows[-1][0]}."
            nums = [r for r in rows if isinstance(r[1], (int, float))]
            if nums:
                peak = max(nums, key=lambda r: r[1])
                text += f" The highest was {peak[0]} at {fmt(peak[1])}."
        else:
            text += "."
    if result.get("truncated"):
        text += f" Showing the first {len(rows)} of {result['row_count']:,} rows."
    return text + filt


def format_response(result: dict, plan: dict) -> dict:
    """Return the success-specific fields of the API response."""
    out = {"result_type": None, "label": None, "value": None, "columns": None,
           "rows": None, "row_count": None, "truncated": None, "chart": None,
           "explanation": None}

    if result["kind"] == "scalar":
        out.update(result_type="number", label=result["label"], value=result["value"])
        out["explanation"] = _explain(plan, result)
        return out

    chart, note = _chart_spec(plan, result)
    out.update(
        result_type="chart" if chart else "table",
        columns=result["columns"], rows=result["rows"],
        row_count=result["row_count"], truncated=result["truncated"], chart=chart,
    )
    out["explanation"] = _explain(plan, result) + note
    return out
