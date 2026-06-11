"""Leakage-safe feature engineering. All lag/rolling features for day D use only
days <= D-1 (rolls) or exactly D-k (lags), via groupby(id).shift before rolling."""
import gc

import numpy as np
import pandas as pd

LAGS = [7, 14, 28, 35]
CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1"]

FEATURES = (
    [f"lag_{l}" for l in LAGS]
    + ["roll_mean_7", "roll_mean_28", "roll_std_28", "roll_max_28"]
    + ["wday", "month", "week", "is_weekend", "snap"]
    + ["sell_price", "price_change_7d", "price_rank_cat", "price_momentum_28"]
    + [f"{c}_enc" for c in CAT_COLS]
)
TARGET = "sales"


def _grouped_roll(df: pd.DataFrame, col: str, window: int, fn: str) -> pd.Series:
    """Rolling stat over a window ending at D-1 (shift(1) first), per id."""
    shifted = df.groupby("id", observed=True)[col].shift(1)
    roll = getattr(
        shifted.groupby(df["id"], observed=True).rolling(window, min_periods=window), fn
    )()
    result = roll.reset_index(level=0, drop=True).astype(np.float32)
    del shifted, roll
    return result


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input: long base table (one row per id per day) sorted arbitrarily.
    Output: feature table with NaN-lag rows dropped. Requires columns:
    id, item_id..state_id, d_int, sales, date, wm_yr_wk, wday, month,
    event_type_1, snap, sell_price.
    Requires exactly one row per (id, d_int) with no missing days within each
    series (positional shifts assume gapless daily data)."""
    df = df.sort_values(["id", "d_int"]).reset_index(drop=True)
    g = df.groupby("id", observed=True)[TARGET]
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l).astype(np.float32)
    del g
    gc.collect()

    df["roll_mean_7"] = _grouped_roll(df, TARGET, 7, "mean")
    df["roll_mean_28"] = _grouped_roll(df, TARGET, 28, "mean")
    df["roll_std_28"] = _grouped_roll(df, TARGET, 28, "std")
    df["roll_max_28"] = _grouped_roll(df, TARGET, 28, "max")
    gc.collect()

    # calendar
    df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(np.int16)
    df["is_weekend"] = (df["wday"] <= 2).astype(np.int8)  # M5: wday 1=Sat, 2=Sun

    # price — cast to float32 immediately
    gp = df.groupby("id", observed=True)["sell_price"]
    df["price_change_7d"] = (df["sell_price"] / gp.shift(7) - 1).astype(np.float32)
    df["price_momentum_28"] = (df["sell_price"] / gp.shift(28) - 1).astype(np.float32)
    del gp
    df["price_rank_cat"] = df.groupby(["store_id", "cat_id", "wm_yr_wk"], observed=True)[
        "sell_price"
    ].rank(pct=True).astype(np.float32)

    # categorical label encoding — use int16 instead of int32 where possible
    for c in CAT_COLS:
        df[f"{c}_enc"] = (
            df[c].astype(str).fillna("none").astype("category").cat.codes.astype(np.int16)
        )
    gc.collect()

    # drop rows where lags are undefined (series start); fill benign price NaNs
    # lag_35 is the binding constraint; roll_std_28 included defensively
    # NOTE: .dropna() already returns a new dataframe; .copy() is unnecessary
    df = df.dropna(subset=[f"lag_{l}" for l in LAGS] + ["roll_std_28"])
    gc.collect()

    df["price_change_7d"] = df["price_change_7d"].fillna(0.0).astype(np.float32)
    df["price_momentum_28"] = df["price_momentum_28"].fillna(0.0).astype(np.float32)
    df["price_rank_cat"] = df["price_rank_cat"].fillna(0.5).astype(np.float32)

    # compact dtypes — cast any remaining float64 columns in FEATURES to float32
    for c in FEATURES:
        if c in df.columns and df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df.reset_index(drop=True)
