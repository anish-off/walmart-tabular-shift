"""Load raw M5 CSVs, melt wide->long, merge calendar + prices, downcast dtypes."""
from pathlib import Path
import numpy as np
import pandas as pd

MAX_DAY = 1913  # plan uses days 1..1913 (years 1-5)

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLS = ["d", "date", "wm_yr_wk", "wday", "month", "year", "event_type_1",
            "snap_CA", "snap_TX", "snap_WI"]


def load_raw(raw_dir: str | Path):
    raw = Path(raw_dir)
    sales = pd.read_csv(raw / "sales_train_evaluation.csv")
    calendar = pd.read_csv(raw / "calendar.csv", usecols=CAL_COLS)
    prices = pd.read_csv(raw / "sell_prices.csv")
    return sales, calendar, prices


def melt_sales(sales: pd.DataFrame, max_day: int = MAX_DAY) -> pd.DataFrame:
    day_cols = [f"d_{i}" for i in range(1, max_day + 1) if f"d_{i}" in sales.columns]
    for c in ID_COLS:
        sales[c] = sales[c].astype("category")
    long = sales.melt(id_vars=ID_COLS, value_vars=day_cols,
                      var_name="d", value_name="sales")
    long["d_int"] = long["d"].str.slice(2).astype(np.int16)
    long["sales"] = long["sales"].astype(np.int16)
    return long.drop(columns=["d"])


def merge_all(long: pd.DataFrame, calendar: pd.DataFrame,
              prices: pd.DataFrame) -> pd.DataFrame:
    calendar = calendar.copy()
    calendar["d_int"] = calendar["d"].str.slice(2).astype(np.int16)
    calendar = calendar.drop(columns=["d"])
    df = long.merge(calendar, on="d_int", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    for col in ["store_id", "item_id"]:
        df[col] = df[col].astype("category")
    df = df[df["sell_price"].notna()]  # rows before product release
    snap = np.select(
        [df["state_id"] == "CA", df["state_id"] == "TX", df["state_id"] == "WI"],
        [df["snap_CA"], df["snap_TX"], df["snap_WI"]], default=0,
    )
    df["snap"] = snap.astype(np.int8)
    df = df.drop(columns=["snap_CA", "snap_TX", "snap_WI"])
    for c, t in [("wday", np.int8), ("month", np.int8), ("year", np.int16),
                 ("wm_yr_wk", np.int32), ("sell_price", np.float32)]:
        df[c] = df[c].astype(t)
    return df.reset_index(drop=True)


def build_base_table(raw_dir: str | Path, sample_items: int | None = None,
                     seed: int = 0) -> pd.DataFrame:
    """Full pipeline: raw CSVs -> merged long table. sample_items keeps a random
    subset of series for smoke tests."""
    sales, calendar, prices = load_raw(raw_dir)
    if sample_items:
        rng = np.random.default_rng(seed)
        keep = rng.choice(len(sales), size=min(sample_items, len(sales)), replace=False)
        sales = sales.iloc[sorted(keep)].reset_index(drop=True)
    return merge_all(melt_sales(sales), calendar, prices)
