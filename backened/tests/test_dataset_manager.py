from app import dataset_manager as dm


def test_list_datasets_skips_missing_files(monkeypatch, tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    registry = {
        "have": {"label": "Have", "file": "a.csv", "description": "present"},
        "missing": {"label": "Missing", "file": "nope.csv", "description": "absent"},
    }
    monkeypatch.setattr(dm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(dm, "_registry", lambda: registry)
    assert [d["name"] for d in dm.list_datasets()] == ["have"]


def test_get_aliases(monkeypatch):
    monkeypatch.setattr(dm, "_registry", lambda: {"hr": {"aliases": {"salary": "MonthlyIncome"}}, "x": {}})
    assert dm.get_aliases("hr") == {"salary": "MonthlyIncome"}
    assert dm.get_aliases("x") == {}
    assert dm.get_aliases("unknown") == {}