import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from shift_study.metrics import per_item_errors


def _make_items(path):
    ids = np.array(["a", "b", "c", "d", "e"])
    y_true = np.ones(5, dtype=np.float32) * 2
    y_pred = np.ones(5, dtype=np.float32) * 2.2
    items = per_item_errors(ids, y_true, y_pred)
    items.to_parquet(path, index=False)


def test_make_report(tmp_path):
    res = tmp_path / "results"
    res.mkdir()

    # fabricate two fake result JSONs
    (res / "lightgbm_e1.json").write_text(json.dumps({
        "model": "lightgbm", "experiment": "e1",
        "wape_mean": 0.55, "wape_std": 0.01,
        "seeds": [42], "wapes_per_seed": [0.55],
    }))
    (res / "lightgbm_e2.json").write_text(json.dumps({
        "model": "lightgbm", "experiment": "e2",
        "wape_mean": 0.63, "wape_std": 0.01,
        "seeds": [42], "wapes_per_seed": [0.63],
    }))

    # fabricate per-item parquets for Wilcoxon path
    _make_items(res / "lightgbm_e1_items.parquet")
    _make_items(res / "lightgbm_e2_items.parquet")

    cmd = [sys.executable, "scripts/make_report.py", "--results", str(res)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    assert (res / "report" / "master_table.csv").exists()
    assert (res / "report" / "report.md").exists()

    mt = pd.read_csv(res / "report" / "master_table.csv")
    lgb_row = mt[mt["model"] == "lightgbm"].iloc[0]
    # delta(0.63, 0.55) = 0.63/0.55 - 1 = 0.14545...
    assert abs(lgb_row["delta_e2"] - 0.1454) < 0.001


def test_make_report_empty(tmp_path):
    """Script must not crash on an empty results directory."""
    res = tmp_path / "empty_results"
    res.mkdir()
    cmd = [sys.executable, "scripts/make_report.py", "--results", str(res)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
