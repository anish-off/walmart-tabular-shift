"""Integration test for scripts/build_features.py using fabricated mini-M5 data."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shift_study.features import FEATURES

# 3 items × 60 days is enough: lag_35 becomes valid at d_36 → min d_int == 36
N_DAYS = 60


def _make_mini_raw(raw_dir: Path) -> None:
    """Write sales_train_evaluation.csv, calendar.csv, sell_prices.csv."""
    rng = np.random.default_rng(42)

    # --- series ids --------------------------------------------------------
    items = [
        ("A_CA_1_evaluation", "A", "FOODS_1", "FOODS", "CA_1", "CA"),
        ("B_TX_1_evaluation", "B", "FOODS_1", "FOODS", "TX_1", "TX"),
        ("C_WI_1_evaluation", "C", "FOODS_1", "FOODS", "WI_1", "WI"),
    ]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [f"d_{i}" for i in range(1, N_DAYS + 1)]
    sales_rows = []
    for row in items:
        sales_row = dict(zip(id_cols, row))
        sales_row.update({f"d_{i}": int(rng.integers(0, 10)) for i in range(1, N_DAYS + 1)})
        sales_rows.append(sales_row)
    sales_df = pd.DataFrame(sales_rows, columns=id_cols + day_cols)
    sales_df.to_csv(raw_dir / "sales_train_evaluation.csv", index=False)

    # --- calendar ----------------------------------------------------------
    # wm_yr_wk: Walmart fiscal week; 7 days per week starting at 11101
    dates = pd.date_range("2011-01-29", periods=N_DAYS)
    wm_yr_wks = [11101 + (i // 7) for i in range(N_DAYS)]
    calendar_df = pd.DataFrame({
        "d": [f"d_{i}" for i in range(1, N_DAYS + 1)],
        "date": dates.astype(str),
        "wm_yr_wk": wm_yr_wks,
        "wday": [(i % 7) + 1 for i in range(N_DAYS)],
        "month": dates.month,
        "year": dates.year,
        "event_type_1": [np.nan] * N_DAYS,
        "snap_CA": rng.integers(0, 2, N_DAYS),
        "snap_TX": rng.integers(0, 2, N_DAYS),
        "snap_WI": rng.integers(0, 2, N_DAYS),
    })
    calendar_df.to_csv(raw_dir / "calendar.csv", index=False)

    # --- sell prices -------------------------------------------------------
    # one price per (store_id, item_id, wm_yr_wk)
    unique_weeks = sorted(set(wm_yr_wks))
    price_rows = []
    for _id, item_id, _dept, _cat, store_id, _state in items:
        for wk in unique_weeks:
            price_rows.append({
                "store_id": store_id,
                "item_id": item_id,
                "wm_yr_wk": wk,
                "sell_price": round(float(rng.uniform(1.0, 5.0)), 2),
            })
    prices_df = pd.DataFrame(price_rows)
    prices_df.to_csv(raw_dir / "sell_prices.csv", index=False)


def test_build_features_script(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    out_path = tmp_path / "features.parquet"

    _make_mini_raw(raw_dir)

    script = Path(__file__).parent.parent / "scripts" / "build_features.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--raw-dir", str(raw_dir),
         "--out", str(out_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"Script failed with exit code {result.returncode}.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert out_path.exists(), "Output parquet file not created"
    df = pd.read_parquet(out_path)

    assert len(df) > 0, "Output parquet has zero rows"

    missing = [c for c in FEATURES if c not in df.columns]
    assert not missing, f"Missing FEATURES columns: {missing}"

    nan_cols = [c for c in FEATURES if df[c].isna().any()]
    assert not nan_cols, f"NaNs found in FEATURES columns: {nan_cols}"

    assert int(df["d_int"].min()) == 36, (
        f"Expected min d_int == 36 (lag_35 binding), got {df['d_int'].min()}"
    )
