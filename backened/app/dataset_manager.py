"""Dataset manager: loads approved CSVs and exposes their schema.

Adding a dataset = adding an entry to data/datasets.json. No code changes.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REGISTRY_PATH = DATA_DIR / "datasets.json"


class DatasetError(Exception):
    """Raised for unknown datasets or unreadable files."""


@lru_cache(maxsize=1)
def _registry() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_datasets() -> list[dict]:
    """Datasets that are registered AND whose CSV file is actually present."""
    return [
        {"name": name, "label": cfg["label"], "description": cfg["description"]}
        for name, cfg in _registry().items()
        if (DATA_DIR / cfg["file"]).exists()
    ]


def _config(name: str) -> dict:
    cfg = _registry().get(name)
    if cfg is None:
        raise DatasetError(f"Unknown dataset '{name}'.")
    return cfg


def get_aliases(name: str) -> dict:
    """Synonyms for column names, e.g. {"salary": "MonthlyIncome"}. Empty if none or unknown."""
    return dict(_registry().get(name, {}).get("aliases", {}))


@lru_cache(maxsize=8)
def load_dataset(name: str) -> pd.DataFrame:
    """Load, clean column names, parse dates, drop useless columns."""
    cfg = _config(name)
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise DatasetError(f"Data file '{cfg['file']}' not found in backend/data/.")

    df = pd.read_csv(path, **cfg.get("read_options", {}))
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in cfg.get("drop_columns", []) if c in df.columns])

    for col, fmt in cfg.get("date_columns", {}).items():
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype("string").str.strip(), format=fmt, errors="coerce")
    return df


def _dtype_label(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "text"


def get_schema(name: str, sample_rows: int = 5) -> dict:
    """Everything the preview panel needs (FR-02)."""
    cfg = _config(name)
    df = load_dataset(name)
    notes = cfg.get("column_notes", {})
    columns = [
        {
            "name": col,
            "type": _dtype_label(df[col]),
            "missing": int(df[col].isna().sum()),
            "note": notes.get(col, ""),
        }
        for col in df.columns
    ]
    sample = df.head(sample_rows).astype(object).where(df.head(sample_rows).notna(), None)
    return {
        "name": name,
        "label": cfg["label"],
        "description": cfg["description"],
        "rows": int(len(df)),
        "column_count": int(df.shape[1]),
        "columns": columns,
        "sample": json.loads(sample.to_json(orient="records", date_format="iso")),
    }


def get_llm_context(name: str) -> str:
    """Schema-only text for the LLM prompt (FR-04). Never includes raw rows."""
    schema = get_schema(name, sample_rows=0)
    lines = [f"Dataset: {schema['label']} - {schema['description']}",
             f"Rows: {schema['rows']}", "Columns:"]
    for c in schema["columns"]:
        note = f" - {c['note']}" if c["note"] else ""
        lines.append(f"- \"{c['name']}\" ({c['type']}){note}")
    return "\n".join(lines)