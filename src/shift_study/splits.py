"""E1 (random), E2 (temporal), E3 (cold-start) split definitions."""
import numpy as np
import pandas as pd

TRAIN_END = 1521       # last training day (years 1-4)
VAL_START = 1400       # validation window: 1400..1521
TEST_START = 1522      # year 5
TEST_END = 1913
COLD_START_MAX_NONZERO = 30


def e1_split(df: pd.DataFrame, seed: int = 42, test_frac: float = 0.2,
             val_frac: float = 0.1):
    """Random 70/10/20 split (train/val/test) over ALL rows (temporal leakage by design —
    E1 is the Delta denominator)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    n_val = int(len(df) * val_frac)
    test = df.iloc[idx[:n_test]]
    val = df.iloc[idx[n_test:n_test + n_val]]
    train = df.iloc[idx[n_test + n_val:]]
    return train, val, test


def e2_split(df: pd.DataFrame):
    """Temporal split. Train < 1400, val 1400..1521 (held-out window for
    tuning/early stopping), test 1522..1913."""
    train = df[df["d_int"] < VAL_START]
    val = df[(df["d_int"] >= VAL_START) & (df["d_int"] <= TRAIN_END)]
    test = df[df["d_int"] >= TEST_START]
    if not train.empty and not test.empty:
        if not (test["d_int"].min() > train["d_int"].max()):
            raise RuntimeError(
                f"Leakage: test min={test['d_int'].min()} <= train max={train['d_int'].max()}"
            )
    return train, val, test


def cold_start_ids(df: pd.DataFrame) -> set:
    """Items with < COLD_START_MAX_NONZERO non-zero sales days in d <= TRAIN_END."""
    hist = df[df["d_int"] <= TRAIN_END]
    nz = hist[hist["sales"] > 0].groupby("id", observed=True)["d_int"].nunique()
    all_ids = hist["id"].unique()
    counts = pd.Series(0, index=all_ids, dtype=int)
    counts.loc[nz.index] = nz
    return set(counts[counts < COLD_START_MAX_NONZERO].index)


def e3_split(df: pd.DataFrame):
    """E2 split restricted to cold-start items."""
    ids = cold_start_ids(df)
    sub = df[df["id"].isin(ids)]
    if sub.empty:
        empty = sub.iloc[0:0]
        return empty, empty, empty
    return e2_split(sub)
