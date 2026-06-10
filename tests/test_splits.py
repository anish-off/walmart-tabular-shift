import numpy as np
import pandas as pd
from shift_study.splits import (
    TRAIN_END, VAL_START, TEST_START, e1_split, e2_split, e3_split, cold_start_ids,
)


def make_df():
    """3 items x days 1..1913. Item 'cold' sells on only 10 days; others daily."""
    rows = []
    for item in ["hot_a", "hot_b", "cold"]:
        for d in range(1, 1914):
            if item == "cold":
                sales = 5 if d in range(100, 110) else 0
            else:
                sales = 1 + d % 3
            rows.append((item, d, sales))
    return pd.DataFrame(rows, columns=["id", "d_int", "sales"])


def test_e1_disjoint_and_proportions():
    df = make_df()
    tr, va, te = e1_split(df, seed=0)
    assert len(tr) + len(va) + len(te) == len(df)
    assert set(tr.index).isdisjoint(te.index) and set(tr.index).isdisjoint(va.index)
    assert abs(len(te) / len(df) - 0.2) < 0.01
    assert abs(len(va) / len(df) - 0.1) < 0.01
    assert set(va.index).isdisjoint(te.index)


def test_e1_deterministic():
    df = make_df()
    _, _, te1 = e1_split(df, seed=0)
    _, _, te2 = e1_split(df, seed=0)
    assert te1.index.equals(te2.index)


def test_e2_temporal_boundaries():
    df = make_df()
    tr, va, te = e2_split(df)
    assert tr["d_int"].max() < VAL_START
    assert va["d_int"].min() == VAL_START and va["d_int"].max() == TRAIN_END
    assert te["d_int"].min() == TEST_START
    assert te["d_int"].min() > tr["d_int"].max()  # leakage guard


def test_cold_start_ids():
    ids = cold_start_ids(make_df())
    assert ids == {"cold"}


def test_e3_only_cold_items():
    tr, va, te = e3_split(make_df())
    for part in (tr, va, te):
        assert set(part["id"].unique()) <= {"cold"}
    assert te["d_int"].min() >= TEST_START
