import numpy as np
import pandas as pd
from shift_study.features import add_features, FEATURES, CAT_COLS


def make_long_df(n_days=80):
    """Two series with deterministic sales and prices, calendar columns present."""
    rows = []
    for sid, item in [("A_CA_1", "A"), ("B_CA_1", "B")]:
        for d in range(1, n_days + 1):
            rows.append({
                "id": sid, "item_id": item, "dept_id": "D1", "cat_id": "FOODS",
                "store_id": "CA_1", "state_id": "CA",
                "d_int": d, "sales": d if sid == "A_CA_1" else 2 * d,
                "date": pd.Timestamp("2011-01-28") + pd.Timedelta(days=d),
                "wm_yr_wk": 11100 + d // 7, "wday": (d % 7) + 1, "month": 1,
                "year": 2011, "event_type_1": np.nan,
                "snap": 0, "sell_price": 1.0 + 0.01 * d,
            })
    return pd.DataFrame(rows)


def test_lag_values_are_strictly_past():
    df = add_features(make_long_df())
    a = df[df["id"] == "A_CA_1"].set_index("d_int")
    assert a.loc[50, "lag_7"] == 43      # sales(43) for series A = 43
    assert a.loc[50, "lag_28"] == 22
    # roll_mean_7 at day 50 = mean(sales days 43..49)
    assert a.loc[50, "roll_mean_7"] == np.mean(range(43, 50))
    assert a.loc[50, "roll_max_28"] == 49


def test_no_current_day_leakage():
    base = make_long_df()
    f1 = add_features(base.copy())
    spiked = base.copy()
    spiked.loc[(spiked["id"] == "A_CA_1") & (spiked["d_int"] == 50), "sales"] = 9999
    f2 = add_features(spiked)
    lag_cols = [c for c in FEATURES if c.startswith(("lag_", "roll_"))]
    r1 = f1[(f1["id"] == "A_CA_1") & (f1["d_int"] == 50)][lag_cols]
    r2 = f2[(f2["id"] == "A_CA_1") & (f2["d_int"] == 50)][lag_cols]
    pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))
    # the day-50 spike IS visible at day 57's lag_7 in the spiked frame
    a2 = f2[(f2["id"] == "A_CA_1") & (f2["d_int"] == 57)]
    assert a2["lag_7"].values[0] == 9999


def test_nan_lag_rows_dropped():
    df = add_features(make_long_df())
    assert df["d_int"].min() == 36  # first 35 days lack lag_35
    assert not df[FEATURES].isna().any().any()


def test_price_change_7d():
    df = add_features(make_long_df())
    a = df[df["id"] == "A_CA_1"].set_index("d_int")
    expected = (1.0 + 0.01 * 50) / (1.0 + 0.01 * 43) - 1
    assert abs(a.loc[50, "price_change_7d"] - expected) < 1e-6


def test_categoricals_encoded():
    df = add_features(make_long_df())
    for c in CAT_COLS:
        assert df[f"{c}_enc"].dtype.kind in "iu"
