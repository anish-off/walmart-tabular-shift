import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")
from shift_study.features import FEATURES


def make_fake_features(path, n_items=8, n_days=1913):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_items):
        sid = f"item{i}_CA_1"
        base = rng.integers(1, 8)
        for d in range(36, n_days + 1):
            rows.append({"id": sid, "d_int": d, "sales": int(base + d % 4),
                         "item_id": f"item{i}", "store_id": "CA_1", "cat_id": "F"})
    df = pd.DataFrame(rows)
    for c in FEATURES:
        df[c] = rng.normal(size=len(df)).astype(np.float32)
    for c in ["item_id_enc", "dept_id_enc", "cat_id_enc", "store_id_enc",
              "state_id_enc", "event_type_1_enc"]:
        df[c] = rng.integers(0, 5, len(df))
    df.to_parquet(path, index=False)


def test_runner_e2_lightgbm(tmp_path):
    feat = tmp_path / "features.parquet"
    make_fake_features(feat)
    out = tmp_path / "results"
    cmd = [sys.executable, "scripts/run_experiment.py", "--model", "lightgbm",
           "--experiment", "e2", "--features", str(feat), "--out", str(out),
           "--config", "configs/lightgbm.yaml"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    payload = json.loads((out / "lightgbm_e2.json").read_text())
    assert payload["experiment"] == "e2"
    assert payload["wape_mean"] > 0
    assert (out / "lightgbm_e2_items.parquet").exists()
