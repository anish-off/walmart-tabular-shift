import numpy as np
import pandas as pd
from shift_study.data import melt_sales, merge_all


def mini_frames():
    sales = pd.DataFrame({
        "id": ["A_CA_1_evaluation", "B_TX_1_evaluation"],
        "item_id": ["A", "B"], "dept_id": ["D1", "D1"], "cat_id": ["F", "F"],
        "store_id": ["CA_1", "TX_1"], "state_id": ["CA", "TX"],
        "d_1": [3, 0], "d_2": [4, 1], "d_3": [0, 2],
    })
    calendar = pd.DataFrame({
        "date": pd.date_range("2011-01-29", periods=3).astype(str),
        "wm_yr_wk": [11101, 11101, 11101],
        "wday": [1, 2, 3], "month": [1, 1, 1], "year": [2011, 2011, 2011],
        "d": ["d_1", "d_2", "d_3"], "event_type_1": [np.nan, "Cultural", np.nan],
        "snap_CA": [1, 0, 0], "snap_TX": [0, 1, 0], "snap_WI": [0, 0, 0],
    })
    prices = pd.DataFrame({
        "store_id": ["CA_1", "TX_1"], "item_id": ["A", "B"],
        "wm_yr_wk": [11101, 11101], "sell_price": [2.5, 1.0],
    })
    return sales, calendar, prices


def test_melt_long_format():
    sales, _, _ = mini_frames()
    long = melt_sales(sales, max_day=3)
    assert len(long) == 6
    assert set(long.columns) >= {"id", "d_int", "sales"}
    row = long[(long["id"] == "A_CA_1_evaluation") & (long["d_int"] == 2)]
    assert row["sales"].item() == 4


def test_merge_adds_calendar_price_and_snap():
    sales, calendar, prices = mini_frames()
    df = merge_all(melt_sales(sales, max_day=3), calendar, prices)
    a2 = df[(df["id"] == "A_CA_1_evaluation") & (df["d_int"] == 2)].iloc[0]
    assert a2["sell_price"] == 2.5
    assert a2["snap"] == 0          # snap_CA on d_2 is 0
    b2 = df[(df["id"] == "B_TX_1_evaluation") & (df["d_int"] == 2)].iloc[0]
    assert b2["snap"] == 1          # snap_TX on d_2 is 1
    assert "wday" in df.columns and "wm_yr_wk" in df.columns


def test_merge_drops_rows_without_price():
    sales, calendar, prices = mini_frames()
    prices = prices[prices["item_id"] != "B"]  # B has no price -> pre-release
    df = merge_all(melt_sales(sales, max_day=3), calendar, prices)
    assert set(df["id"].unique()) == {"A_CA_1_evaluation"}
