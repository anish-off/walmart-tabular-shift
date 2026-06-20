"""Run TabM on real data.

Two modes:

  --format walmart   Raw M5 CSVs (same structure as training data)
  --format generic   Any retail CSV: date, item_id, store_id, sales, sell_price

Examples
--------
Walmart M5 format:
  python scripts/predict_real.py \\
      --model-path results/tabm_e2_seed42.pt \\
      --format walmart \\
      --raw-dir /home/23alr106/code/walmart-tabular-shift/data/raw

Generic retail CSV:
  python scripts/predict_real.py \\
      --model-path results/tabm_e2_seed42.pt \\
      --format generic \\
      --csv my_sales.csv \\
      --out predictions.csv

Required columns for --format generic:
  date         YYYY-MM-DD
  item_id      any string  (e.g. "MILK_1")
  store_id     any string  (e.g. "STORE_A")
  sales        number
  sell_price   float

Optional columns (defaults used if absent):
  dept_id, cat_id, state_id   → "UNKNOWN"
  event_type_1                → ""  (no event)
  snap                        → 0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from shift_study.features import FEATURES, TARGET, add_features
from shift_study.models.tabm import TabMModel


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_walmart(raw_dir: str) -> pd.DataFrame:
    from shift_study.data import build_base_table
    print("Loading M5 raw CSVs...")
    base = build_base_table(raw_dir)
    print(f"  {len(base):,} rows  |  {base['id'].nunique():,} series")
    print("Building features...")
    df = add_features(base)
    print(f"  {len(df):,} rows after lag-drop")
    return df


def _build_generic(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, parse_dates=["date"])

    required = {"date", "item_id", "store_id", "sales", "sell_price"}
    missing = required - set(raw.columns)
    if missing:
        sys.exit(f"CSV is missing required columns: {missing}")

    # defaults for optional columns
    for col, default in [
        ("dept_id", "UNKNOWN"), ("cat_id", "UNKNOWN"),
        ("state_id", "UNKNOWN"), ("event_type_1", ""), ("snap", 0),
    ]:
        if col not in raw.columns:
            raw[col] = default

    # create unique series id and integer day index
    raw["id"] = raw["item_id"].astype(str) + "_" + raw["store_id"].astype(str)
    raw = raw.sort_values(["id", "date"]).reset_index(drop=True)
    date_origin = raw["date"].min()
    raw["d_int"] = (raw["date"] - date_origin).dt.days.astype("int16") + 1

    # calendar columns expected by add_features
    raw["wday"]     = raw["date"].dt.dayofweek.astype("int8")   # 0=Mon
    raw["month"]    = raw["date"].dt.month.astype("int8")
    # ISO week packed as YYYYWW int
    iso = raw["date"].dt.isocalendar()
    raw["wm_yr_wk"] = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).astype("int32")

    raw["sales"]      = raw["sales"].astype("float32")
    raw["sell_price"] = raw["sell_price"].astype("float32")
    raw["snap"]       = raw["snap"].astype("int8")

    n_series = raw["id"].nunique()
    print(f"Loaded {len(raw):,} rows  |  {n_series:,} unique series")
    if n_series < 2:
        print("  ⚠️  Only one series — lag/rolling features need at least 36+ days per series.")

    print("Building features...")
    df = add_features(raw)
    if df.empty:
        sys.exit(
            "No rows remain after feature engineering.\n"
            "Each series needs at least 35 consecutive days with sales data "
            "(lag_35 is the binding constraint)."
        )
    print(f"  {len(df):,} rows after lag-drop  |  {df['id'].nunique():,} series")
    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True,
                    help="Path to tabm_*.pt checkpoint")
    ap.add_argument("--format", required=True, choices=["walmart", "generic"],
                    help="'walmart' = M5 raw CSVs | 'generic' = your own retail CSV")
    ap.add_argument("--raw-dir", default="data/raw",
                    help="[walmart] directory with M5 CSVs")
    ap.add_argument("--csv",
                    help="[generic] path to your retail CSV")
    ap.add_argument("--out", default="predictions.csv",
                    help="output CSV with a 'pred' column")
    args = ap.parse_args()

    if args.format == "walmart":
        df = _build_walmart(args.raw_dir)
    else:
        if not args.csv:
            sys.exit("--csv is required for --format generic")
        df = _build_generic(args.csv)

    # load model
    print(f"\nLoading model from {args.model_path} ...")
    model = TabMModel.load(args.model_path)

    # predict
    print("Running predictions...")
    preds = model.predict(df[FEATURES])

    # output
    out_df = df[["id", "d_int"]].copy()
    if TARGET in df.columns:
        out_df["actual"] = df[TARGET].values
        actual = df[TARGET].to_numpy(dtype=np.float32)
        total_actual = actual.sum()
        wape = float(np.abs(actual - preds).sum() / total_actual) if total_actual > 0 else float("nan")
        print(f"\nWAPE = {wape:.4f}")
    out_df["pred"] = preds

    out_df.to_csv(args.out, index=False)
    print(f"Predictions saved -> {args.out}  ({len(out_df):,} rows)")


if __name__ == "__main__":
    main()
