# Tabular Shift Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full codebase for the M5 temporal-shift benchmark: 5 models (XGBoost, LightGBM, TabPFN v2, TabM, TabR) × 3 experiments (E1 random, E2 temporal, E3 cold-start), WAPE/Delta metrics, crossover curve, and report generation.

**Architecture:** Python package `shift_study` (src layout) with shared data/features/splits/metrics modules and a common model interface; thin CLI scripts drive one-time feature building and per-(model, experiment) runs that each write JSON + per-item-error parquet immediately. Target environment: Linux GPU server; smoke-testable anywhere via `--sample`.

**Tech Stack:** Python ≥3.10, pandas/numpy/pyarrow, scikit-learn, scipy, xgboost, lightgbm, tabpfn (v2), PyTorch (TabM & TabR implemented from papers), optuna, matplotlib, pyyaml, kaggle CLI, pytest.

**Conventions used throughout:**
- `id` column = item×store series key (e.g. `FOODS_3_090_CA_1_evaluation`).
- Day numbering: `d_int` 1…1913. Constants: `TRAIN_END=1521`, `VAL_START=1400`, `TEST_START=1522`, `TEST_END=1913`, cold-start = `<30` non-zero sales days in `d_int ≤ 1521`.
- E2/E3 final-model training uses `d_int < 1400`; days 1400–1521 are the held-out validation window (early stopping + tuning); test never touched until final evaluation.
- All NaN-lag rows (first 35 days of each series) are dropped once at feature-build time, so every model sees identical rows.

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `.gitignore`, `src/shift_study/__init__.py`, `src/shift_study/models/__init__.py`, `configs/xgboost.yaml`, `configs/lightgbm.yaml`, `configs/tabpfn.yaml`, `configs/tabm.yaml`, `configs/tabr.yaml`, `tests/__init__.py`

- [ ] **Step 1: Write all scaffolding files**

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "shift-study"
version = "0.1.0"
description = "M5 temporal distribution shift study: trees vs tabular deep learning"
requires-python = ">=3.10"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`requirements.txt`:
```
numpy
pandas>=2.0
pyarrow
scikit-learn
scipy
xgboost>=2.0
lightgbm>=4.0
tabpfn>=2.0
torch>=2.1
optuna
matplotlib
pyyaml
kaggle
pytest
```

`.gitignore`:
```
__pycache__/
*.egg-info/
.pytest_cache/
data/
results/
*.parquet
*.zip
```

`src/shift_study/__init__.py` and `src/shift_study/models/__init__.py` and `tests/__init__.py`: empty.

`configs/xgboost.yaml`:
```yaml
model: xgboost
seeds: [42]
early_stopping_rounds: 50
params:
  n_estimators: 1000
  learning_rate: 0.05
  max_depth: 7
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 10
  tree_method: hist
  objective: reg:absoluteerror
```

`configs/lightgbm.yaml`:
```yaml
model: lightgbm
seeds: [42]
early_stopping_rounds: 75
params:
  objective: regression_l1
  n_estimators: 1500
  learning_rate: 0.05
  num_leaves: 127
  min_child_samples: 50
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 1
  lambda_l2: 1.0
  verbosity: -1
```

`configs/tabpfn.yaml`:
```yaml
model: tabpfn
seeds: [0, 1, 2, 3, 4]
params:
  context_rows: 8000
  predict_chunk: 50000
  device: cuda
```

`configs/tabm.yaml`:
```yaml
model: tabm
seeds: [42]
params:
  d_main: 512
  k: 8
  n_blocks: 3
  dropout: 0.0
  lr: 1.0e-3
  weight_decay: 1.0e-5
  batch_size: 4096
  max_epochs: 200
  patience: 20
  device: cuda
```

`configs/tabr.yaml`:
```yaml
model: tabr
seeds: [42]
params:
  d_main: 96
  context_size: 96
  dropout: 0.1
  lr: 5.0e-4
  weight_decay: 1.0e-5
  batch_size: 1024
  max_epochs: 100
  patience: 10
  context_freeze_epoch: 4
  train_subsample: 500000
  candidate_chunk: 65536
  device: cuda
```

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "chore: scaffold shift-study project"
```

---

### Task 2: Metrics module (TDD)

**Files:**
- Create: `src/shift_study/metrics.py`
- Test: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from shift_study.metrics import (
    wape, delta, shift_sensitivity, per_item_errors, wilcoxon_items,
)


def test_wape_known_value():
    # |10-8| + |0-1| + |5-5| = 3 ; sum actual = 15
    assert wape([10, 0, 5], [8, 1, 5]) == pytest.approx(3 / 15)


def test_wape_perfect():
    assert wape([3, 4], [3, 4]) == 0.0


def test_wape_zero_denominator_is_nan():
    assert np.isnan(wape([0, 0], [1, 2]))


def test_delta():
    assert delta(0.63, 0.55) == pytest.approx(0.63 / 0.55 - 1)


def test_shift_sensitivity():
    assert shift_sensitivity(0.05, 0.10) == pytest.approx(0.5)


def test_per_item_errors_aggregates_by_id():
    ids = ["a", "a", "b"]
    out = per_item_errors(ids, [10, 0, 5], [8, 1, 5])
    out = out.set_index("id")
    assert out.loc["a", "sum_abs_err"] == 3
    assert out.loc["a", "sum_actual"] == 10
    assert out.loc["a", "n"] == 2
    assert out.loc["b", "sum_abs_err"] == 0


def test_wilcoxon_items_detects_better_model():
    rng = np.random.default_rng(0)
    ids = [f"i{k}" for k in range(50)]
    actual = rng.integers(1, 20, 50)
    err_a = per_item_errors(ids, actual, actual + rng.normal(0, 0.5, 50))
    err_b = per_item_errors(ids, actual, actual + rng.normal(0, 5.0, 50))
    stat, p = wilcoxon_items(err_a, err_b)
    assert p < 0.01  # model a clearly better
```

- [ ] **Step 2: Run tests, verify they fail** — `pytest tests/test_metrics.py -q` → ImportError.

- [ ] **Step 3: Implement `src/shift_study/metrics.py`**

```python
"""Evaluation metrics: WAPE, Delta, shift sensitivity, per-item errors, Wilcoxon."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error: sum|err| / sum|actual|."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def delta(wape_shift: float, wape_base: float) -> float:
    """Shift degradation: (WAPE_shift / WAPE_base) - 1."""
    return float(wape_shift / wape_base - 1.0)


def shift_sensitivity(delta_e3: float, delta_e2: float) -> float:
    """Delta_E3 normalized by Delta_E2."""
    if delta_e2 == 0:
        return float("nan")
    return float(delta_e3 / delta_e2)


def per_item_errors(ids, y_true, y_pred) -> pd.DataFrame:
    """Aggregate absolute error and actuals per item id (enables item-level
    WAPE, Wilcoxon tests, and any-level re-aggregation without re-running models)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    df = pd.DataFrame({
        "id": np.asarray(ids),
        "abs_err": np.abs(y_true - y_pred),
        "actual": np.abs(y_true),
    })
    return (
        df.groupby("id", observed=True)
        .agg(sum_abs_err=("abs_err", "sum"), sum_actual=("actual", "sum"), n=("abs_err", "size"))
        .reset_index()
    )


def item_wape(per_item: pd.DataFrame) -> pd.Series:
    """Per-item WAPE; items with zero actuals -> NaN."""
    denom = per_item["sum_actual"].replace(0, np.nan)
    return per_item["sum_abs_err"] / denom


def wilcoxon_items(err_a: pd.DataFrame, err_b: pd.DataFrame) -> tuple[float, float]:
    """Wilcoxon signed-rank on per-item WAPE of model A vs model B."""
    m = err_a.merge(err_b, on="id", suffixes=("_a", "_b"))
    wa = m["sum_abs_err_a"] / m["sum_actual_a"].replace(0, np.nan)
    wb = m["sum_abs_err_b"] / m["sum_actual_b"].replace(0, np.nan)
    mask = wa.notna() & wb.notna() & (wa != wb)
    stat, p = wilcoxon(wa[mask], wb[mask])
    return float(stat), float(p)
```

- [ ] **Step 4: Run tests, verify pass** — `pytest tests/test_metrics.py -q` → all pass.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: metrics (WAPE, Delta, shift sensitivity, Wilcoxon)"`

---

### Task 3: Splits module (TDD)

**Files:**
- Create: `src/shift_study/splits.py`
- Test: `tests/test_splits.py`

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_splits.py -q` → ImportError.

- [ ] **Step 3: Implement `src/shift_study/splits.py`**

```python
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
    """Random 80/20 split over ALL rows (leakage by design — Delta denominator).
    A random val_frac of the remainder is held out for early stopping."""
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
    assert test["d_int"].min() > train["d_int"].max()
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
    return e2_split(sub)
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_splits.py -q`

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: E1/E2/E3 split logic"`

---

### Task 4: Feature engineering (TDD, leakage-safe)

**Files:**
- Create: `src/shift_study/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing tests**

```python
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


def test_nan_lag_rows_dropped():
    df = add_features(make_long_df())
    assert df["d_int"].min() == 36  # first 35 days lack lag_35
    assert not df[FEATURES].isna().any().any()


def test_price_change_7d():
    df = add_features(make_long_df())
    a = df[df["id"] == "A_CA_1"].set_index("d_int")
    expected = (1.0 + 0.01 * 50) / (1.0 + 0.01 * 43) - 1
    assert abs(a.loc[50, "price_change_7d"] - expected) < 1e-9


def test_categoricals_encoded():
    df = add_features(make_long_df())
    for c in CAT_COLS:
        assert df[f"{c}_enc"].dtype.kind in "iu"
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `src/shift_study/features.py`**

```python
"""Leakage-safe feature engineering. All lag/rolling features for day D use only
days <= D-1 (rolls) or exactly D-k (lags), via groupby(id).shift before rolling."""
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
    return roll.reset_index(level=0, drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input: long base table (one row per id per day) sorted arbitrarily.
    Output: feature table with NaN-lag rows dropped. Requires columns:
    id, item_id..state_id, d_int, sales, date, wm_yr_wk, wday, month,
    event_type_1, snap, sell_price."""
    df = df.sort_values(["id", "d_int"]).reset_index(drop=True)
    g = df.groupby("id", observed=True)[TARGET]
    for l in LAGS:
        df[f"lag_{l}"] = g.shift(l)
    df["roll_mean_7"] = _grouped_roll(df, TARGET, 7, "mean")
    df["roll_mean_28"] = _grouped_roll(df, TARGET, 28, "mean")
    df["roll_std_28"] = _grouped_roll(df, TARGET, 28, "std")
    df["roll_max_28"] = _grouped_roll(df, TARGET, 28, "max")

    # calendar
    df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(np.int16)
    df["is_weekend"] = (df["wday"] <= 2).astype(np.int8)  # M5: wday 1=Sat, 2=Sun

    # price
    gp = df.groupby("id", observed=True)["sell_price"]
    df["price_change_7d"] = df["sell_price"] / gp.shift(7) - 1
    df["price_momentum_28"] = df["sell_price"] / gp.shift(28) - 1
    df["price_rank_cat"] = df.groupby(["store_id", "cat_id", "wm_yr_wk"], observed=True)[
        "sell_price"
    ].rank(pct=True)

    # categorical label encoding
    for c in CAT_COLS:
        df[f"{c}_enc"] = (
            df[c].astype(str).fillna("none").astype("category").cat.codes.astype(np.int32)
        )

    # drop rows where lags are undefined (series start); fill benign price NaNs
    df = df.dropna(subset=[f"lag_{l}" for l in LAGS] + ["roll_std_28"])
    df["price_change_7d"] = df["price_change_7d"].fillna(0.0)
    df["price_momentum_28"] = df["price_momentum_28"].fillna(0.0)
    df["price_rank_cat"] = df["price_rank_cat"].fillna(0.5)

    # compact dtypes
    for c in FEATURES:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_features.py -q`

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: leakage-safe feature engineering"`

---

### Task 5: Data loading + download script

**Files:**
- Create: `src/shift_study/data.py`, `scripts/download_data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests** (synthetic mini-M5 frames, no real data needed)

```python
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
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `src/shift_study/data.py`**

```python
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
```

- [ ] **Step 4: Implement `scripts/download_data.py`**

```python
"""Download M5 data via the Kaggle API into data/raw/.
Requires ~/.kaggle/kaggle.json (Kaggle > Account > Create API Token)."""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

COMPETITION = "m5-forecasting-accuracy"
NEEDED = ["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if all((out / f).exists() for f in NEEDED):
        print("All M5 files already present, nothing to do.")
        return
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(out)]
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit("kaggle download failed - is kaggle.json configured and the "
                 "competition rules accepted on the website?")
    for z in out.glob("*.zip"):
        print("Extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        z.unlink()
    missing = [f for f in NEEDED if not (out / f).exists()]
    if missing:
        sys.exit(f"Missing after download: {missing}")
    print("Done. Files in", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, verify pass** — `pytest tests/test_data.py -q`

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: M5 data loading and download script"`

---

### Task 6: Feature build script

**Files:**
- Create: `scripts/build_features.py`

- [ ] **Step 1: Implement**

```python
"""One-time feature build: raw CSVs -> data/processed/features.parquet.
Every model consumes this exact table. Run with --sample 500 for a smoke test."""
import argparse
import time
from pathlib import Path

from shift_study.data import build_base_table
from shift_study.features import add_features, FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/processed/features.parquet")
    ap.add_argument("--sample", type=int, default=None,
                    help="keep only N random series (smoke test)")
    args = ap.parse_args()

    t0 = time.time()
    print("Building base table...")
    base = build_base_table(args.raw_dir, sample_items=args.sample)
    print(f"  rows={len(base):,}  series={base['id'].nunique():,}  "
          f"days={base['d_int'].min()}..{base['d_int'].max()}")

    print("Adding features...")
    df = add_features(base)
    keep = ["id", "d_int", "sales", "item_id", "store_id", "cat_id"] + FEATURES
    df = df[keep]
    print(f"  rows after NaN-lag drop={len(df):,}  features={len(FEATURES)}")
    assert not df[FEATURES].isna().any().any(), "NaNs remain in feature columns"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.0f} MB) "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test** — `python scripts/build_features.py --sample 50` (needs real data; on machines without data this is deferred to the server). Expected output: row counts, parquet written.

- [ ] **Step 3: Commit** — `git add -A && git commit -m "feat: feature build script"`

---

### Task 7: Model interface + tree baselines (XGBoost, LightGBM)

**Files:**
- Create: `src/shift_study/models/base.py`, `src/shift_study/models/xgb.py`, `src/shift_study/models/lgbm.py`
- Test: `tests/test_models_trees.py`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pandas as pd
import pytest
from shift_study.metrics import wape
from shift_study.models.base import get_model


def synth(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
        "f3": rng.integers(0, 5, n).astype(float),
    })
    y = 10 + 3 * X["f1"] - 2 * X["f2"] + X["f3"] + rng.normal(0, 0.3, n)
    return X, y.to_numpy()


@pytest.mark.parametrize("name,params", [
    ("xgboost", {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4,
                 "tree_method": "hist", "objective": "reg:absoluteerror"}),
    ("lightgbm", {"objective": "regression_l1", "n_estimators": 200,
                  "learning_rate": 0.1, "num_leaves": 31, "verbosity": -1}),
])
def test_tree_models_learn(name, params):
    X, y = synth()
    m = get_model(name, {"params": params, "early_stopping_rounds": 20}, seed=42)
    m.fit(X[:3000], y[:3000], X[3000:3500], y[3000:3500])
    pred = m.predict(X[3500:])
    baseline = wape(y[3500:], np.full(500, y[:3000].mean()))
    assert wape(y[3500:], pred) < baseline * 0.3
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement `src/shift_study/models/base.py`**

```python
"""Common model interface. Every model: fit(X, y, X_val, y_val) / predict(X).
X is a DataFrame of the shared FEATURES columns (all numeric)."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TabularModel(ABC):
    name: str = "base"

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.params = dict(config.get("params", {}))
        self.seed = seed

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray) -> "TabularModel": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def get_model(name: str, config: dict, seed: int = 42) -> TabularModel:
    """Registry with lazy imports so missing heavy deps only break their model."""
    if name == "xgboost":
        from .xgb import XGBModel
        return XGBModel(config, seed)
    if name == "lightgbm":
        from .lgbm import LGBMModel
        return LGBMModel(config, seed)
    if name == "tabpfn":
        from .tabpfn import TabPFNModel
        return TabPFNModel(config, seed)
    if name == "tabm":
        from .tabm import TabMModel
        return TabMModel(config, seed)
    if name == "tabr":
        from .tabr import TabRModel
        return TabRModel(config, seed)
    raise ValueError(f"Unknown model: {name}")
```

- [ ] **Step 4: Implement `src/shift_study/models/xgb.py`**

```python
import numpy as np
import xgboost as xgb

from .base import TabularModel


class XGBModel(TabularModel):
    name = "xgboost"

    def fit(self, X, y, X_val, y_val):
        self.model = xgb.XGBRegressor(
            **self.params,
            random_state=self.seed,
            early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
            n_jobs=-1,
        )
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=100)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X))
```

- [ ] **Step 5: Implement `src/shift_study/models/lgbm.py`**

```python
import lightgbm as lgb
import numpy as np

from .base import TabularModel


class LGBMModel(TabularModel):
    name = "lightgbm"

    def fit(self, X, y, X_val, y_val):
        self.model = lgb.LGBMRegressor(**self.params, random_state=self.seed, n_jobs=-1)
        self.model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(self.config.get("early_stopping_rounds", 75)),
                       lgb.log_evaluation(100)],
        )
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X))
```

- [ ] **Step 6: Run, verify pass** — `pytest tests/test_models_trees.py -q` (skips automatically if xgboost/lightgbm not installed locally — add `pytest.importorskip("xgboost")` / `("lightgbm")` at module top if needed).

- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat: model interface + XGBoost/LightGBM wrappers"`

---

### Task 8: TabPFN v2 wrapper

**Files:**
- Create: `src/shift_study/models/tabpfn.py`
- Test: `tests/test_models_tabpfn.py`

- [ ] **Step 1: Write test** (skips if tabpfn not installed)

```python
import numpy as np
import pandas as pd
import pytest

tabpfn = pytest.importorskip("tabpfn")
from shift_study.metrics import wape
from shift_study.models.base import get_model


def test_tabpfn_fit_predict_small():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=600), "f2": rng.normal(size=600)})
    y = 5 + 2 * X["f1"].to_numpy() + rng.normal(0, 0.2, 600)
    m = get_model("tabpfn", {"params": {"context_rows": 400, "predict_chunk": 100,
                                        "device": "cpu"}}, seed=0)
    m.fit(X[:500], y[:500], None, None)
    pred = m.predict(X[500:])
    baseline = wape(y[500:], np.full(100, y[:500].mean()))
    assert wape(y[500:], pred) < baseline
```

- [ ] **Step 2: Implement `src/shift_study/models/tabpfn.py`**

```python
"""TabPFN v2: in-context learning, no training loop. fit() samples a context
window (default 8,000 rows) with the run's seed; different seeds give different
contexts -> run 5 seeds and report mean/std (handled by run_experiment)."""
import numpy as np

from .base import TabularModel


class TabPFNModel(TabularModel):
    name = "tabpfn"

    def fit(self, X, y, X_val=None, y_val=None):
        from tabpfn import TabPFNRegressor

        n_ctx = int(self.params.get("context_rows", 8000))
        rng = np.random.default_rng(self.seed)
        if len(X) > n_ctx:
            idx = rng.choice(len(X), size=n_ctx, replace=False)
            X_ctx, y_ctx = X.iloc[idx], np.asarray(y)[idx]
        else:
            X_ctx, y_ctx = X, np.asarray(y)
        self.model = TabPFNRegressor(
            device=self.params.get("device", "cuda"),
            random_state=self.seed,
            ignore_pretraining_limits=True,
        )
        self.model.fit(X_ctx, y_ctx)
        return self

    def predict(self, X) -> np.ndarray:
        chunk = int(self.params.get("predict_chunk", 50000))
        out = []
        for start in range(0, len(X), chunk):
            out.append(np.asarray(self.model.predict(X.iloc[start:start + chunk])))
        return np.concatenate(out)
```

- [ ] **Step 3: Run test** — `pytest tests/test_models_tabpfn.py -q` (CPU, small; skipped where tabpfn missing).

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: TabPFN v2 wrapper (context sampling + chunked predict)"`

---

### Task 9: TabM (PyTorch, BatchEnsemble MLP per ICLR 2025)

**Files:**
- Create: `src/shift_study/models/torch_common.py`, `src/shift_study/models/tabm.py`
- Test: `tests/test_models_torch.py` (TabM part)

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
from shift_study.metrics import wape
from shift_study.models.base import get_model


def synth(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "f1": rng.normal(size=n), "f2": rng.normal(size=n),
        "item_id_enc": rng.integers(0, 20, n),
    })
    item_effect = (X["item_id_enc"] % 5).to_numpy() * 2.0
    y = 10 + 3 * X["f1"].to_numpy() - X["f2"].to_numpy() + item_effect \
        + rng.normal(0, 0.3, n)
    return X, y


def test_tabm_learns():
    X, y = synth()
    m = get_model("tabm", {"params": {
        "d_main": 64, "k": 4, "n_blocks": 2, "dropout": 0.0, "lr": 1e-3,
        "weight_decay": 1e-5, "batch_size": 512, "max_epochs": 60,
        "patience": 10, "device": "cpu"}}, seed=0)
    m.fit(X[:4500], y[:4500], X[4500:5000], y[4500:5000])
    pred = m.predict(X[5000:])
    baseline = wape(y[5000:], np.full(1000, y[:4500].mean()))
    assert wape(y[5000:], pred) < baseline * 0.5
```

- [ ] **Step 2: Implement `src/shift_study/models/torch_common.py`** (shared by TabM and TabR)

```python
"""Shared helpers for the PyTorch models: column typing, standardization,
embedding sizing, tensor conversion, early stopping."""
import copy

import numpy as np
import pandas as pd
import torch

# Categorical-encoded columns (integer codes from features.py). Everything else
# in X is treated as numeric and standardized.
CAT_ENC_COLS = ["item_id_enc", "dept_id_enc", "cat_id_enc", "store_id_enc",
                "state_id_enc", "event_type_1_enc"]


def split_columns(X: pd.DataFrame):
    cat_cols = [c for c in CAT_ENC_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def emb_dim(cardinality: int) -> int:
    return int(min(64, max(4, round(1.6 * cardinality ** 0.56))))


class Preprocessor:
    """Standardize numerics (train stats); pass through cat codes with
    cardinalities; unseen/val codes clipped into range."""

    def fit(self, X: pd.DataFrame):
        self.num_cols, self.cat_cols = split_columns(X)
        num = X[self.num_cols].to_numpy(dtype=np.float32)
        self.mean = num.mean(axis=0)
        self.std = num.std(axis=0) + 1e-6
        self.cards = [int(X[c].max()) + 2 for c in self.cat_cols]  # +1 safety slot
        return self

    def transform(self, X: pd.DataFrame, device: str):
        num = (X[self.num_cols].to_numpy(dtype=np.float32) - self.mean) / self.std
        x_num = torch.tensor(num, device=device)
        if self.cat_cols:
            cat = X[self.cat_cols].to_numpy(dtype=np.int64)
            cat = np.clip(cat, 0, np.array(self.cards) - 1)
            x_cat = torch.tensor(cat, device=device)
        else:
            x_cat = torch.zeros((len(X), 0), dtype=torch.long, device=device)
        return x_num, x_cat


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.best_state = None
        self.bad_epochs = 0

    def step(self, metric: float, model: torch.nn.Module) -> bool:
        """Returns True when training should stop."""
        if metric < self.best:
            self.best = metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
```

- [ ] **Step 3: Implement `src/shift_study/models/tabm.py`**

```python
"""TabM (ICLR 2025): parameter-efficient ensemble of k MLPs sharing one backbone.
BatchEnsemble-style: shared weight matrix W per layer plus per-member rank-1
adapters (r_i input scale, s_i output scale, b_i bias), random-sign initialized.
Prediction = mean over the k members. Loss = mean member MAE."""
import numpy as np
import torch
import torch.nn as nn

from .base import TabularModel
from .torch_common import EarlyStopper, Preprocessor, emb_dim


def _random_sign(*shape):
    return torch.where(torch.rand(*shape) < 0.5, -1.0, 1.0)


class BatchEnsembleLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.r = nn.Parameter(_random_sign(k, d_in))
        self.s = nn.Parameter(_random_sign(k, d_out))
        self.bias = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):  # x: (B, k, d_in)
        return ((x * self.r) @ self.weight) * self.s + self.bias


class TabMNet(nn.Module):
    def __init__(self, n_num: int, cards: list[int], d_main: int, k: int,
                 n_blocks: int, dropout: float):
        super().__init__()
        self.k = k
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c)) for c in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        self.blocks = nn.ModuleList()
        d = d_in
        for _ in range(n_blocks):
            self.blocks.append(BatchEnsembleLinear(d, d_main, k))
            d = d_main
        self.drop = nn.Dropout(dropout)
        self.head = BatchEnsembleLinear(d_main, 1, k)

    def forward(self, x_num, x_cat):  # returns (B, k)
        parts = [x_num] + [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        z = torch.cat(parts, dim=1)
        z = z.unsqueeze(1).expand(-1, self.k, -1)
        for blk in self.blocks:
            z = self.drop(torch.relu(blk(z)))
        return self.head(z).squeeze(-1)


class TabMModel(TabularModel):
    name = "tabm"

    def fit(self, X, y, X_val, y_val):
        p = self.params
        device = p.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.prep = Preprocessor().fit(X)
        xn, xc = self.prep.transform(X, "cpu")
        y_t = torch.tensor(np.asarray(y, dtype=np.float32))
        vxn, vxc = self.prep.transform(X_val, device)
        vy = torch.tensor(np.asarray(y_val, dtype=np.float32), device=device)

        self.net = TabMNet(
            n_num=xn.shape[1], cards=self.prep.cards,
            d_main=int(p.get("d_main", 512)), k=int(p.get("k", 8)),
            n_blocks=int(p.get("n_blocks", 3)), dropout=float(p.get("dropout", 0.0)),
        ).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(p.get("lr", 1e-3)),
                                weight_decay=float(p.get("weight_decay", 1e-5)))
        stopper = EarlyStopper(int(p.get("patience", 20)))
        bs = int(p.get("batch_size", 4096))

        n = len(xn)
        for epoch in range(int(p.get("max_epochs", 200))):
            self.net.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                xb_n, xb_c = xn[b].to(device), xc[b].to(device)
                yb = y_t[b].to(device)
                pred = self.net(xb_n, xb_c)               # (B, k)
                loss = (pred - yb.unsqueeze(1)).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val_pred = self._predict_tensor(vxn, vxc, bs)
                val_mae = (val_pred - vy).abs().mean().item()
            if stopper.step(val_mae, self.net):
                break
        stopper.restore(self.net)
        self.device, self.bs = device, bs
        return self

    def _predict_tensor(self, xn, xc, bs):
        out = []
        for s in range(0, len(xn), bs):
            out.append(self.net(xn[s:s + bs], xc[s:s + bs]).mean(dim=1))
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        self.net.eval()
        xn, xc = self.prep.transform(X, self.device)
        return self._predict_tensor(xn, xc, self.bs).cpu().numpy()
```

- [ ] **Step 4: Run test** — `pytest tests/test_models_torch.py -q -k tabm`

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: TabM (BatchEnsemble MLP) in PyTorch"`

---

### Task 10: TabR (PyTorch, retrieval-augmented per ICLR 2024, TabR-S variant)

**Files:**
- Create: `src/shift_study/models/tabr.py`
- Test: add TabR test to `tests/test_models_torch.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_models_torch.py`)

```python
def test_tabr_learns():
    X, y = synth(n=3000, seed=1)
    m = get_model("tabr", {"params": {
        "d_main": 32, "context_size": 16, "dropout": 0.0, "lr": 1e-3,
        "weight_decay": 1e-5, "batch_size": 256, "max_epochs": 40,
        "patience": 8, "context_freeze_epoch": 2, "train_subsample": 2000,
        "candidate_chunk": 1024, "device": "cpu"}}, seed=0)
    m.fit(X[:2200], y[:2200], X[2200:2500], y[2200:2500])
    pred = m.predict(X[2500:])
    baseline = wape(y[2500:], np.full(500, y[:2200].mean()))
    assert wape(y[2500:], pred) < baseline * 0.6
```

- [ ] **Step 2: Implement `src/shift_study/models/tabr.py`**

```python
"""TabR-S (ICLR 2024): feed-forward encoder + differentiable k-NN retrieval.
For each query x: encode h=E(x), key k_x=K(h); retrieve top-m training candidates
by -||k_x - k_i||^2; context value v_i = Wy(y_i) + T(k_i - k_x); output =
head(h + sum_i softmax(sim)_i * v_i).

M5-scale concessions from the plan: candidates = 500K-row training subsample,
candidate keys frozen after `context_freeze_epoch` epochs, chunked top-m search.
Self-retrieval is masked during training (a training row must not retrieve its
own label)."""
import numpy as np
import torch
import torch.nn as nn

from .base import TabularModel
from .torch_common import EarlyStopper, Preprocessor, emb_dim


class TabRNet(nn.Module):
    def __init__(self, n_num: int, cards: list[int], d: int, dropout: float):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim(c)) for c in cards])
        d_in = n_num + sum(e.embedding_dim for e in self.embs)
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(),
                                 nn.Dropout(dropout), nn.Linear(d, d))
        self.key = nn.Linear(d, d)
        self.label_emb = nn.Linear(1, d)
        self.t = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                               nn.Dropout(dropout), nn.Linear(d, d))
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(d, 1))
        self.scale = d ** -0.5

    def featurize(self, x_num, x_cat):
        parts = [x_num] + [e(x_cat[:, i]) for i, e in enumerate(self.embs)]
        return torch.cat(parts, dim=1)

    def encode(self, x_num, x_cat):
        return self.enc(self.featurize(x_num, x_cat))

    def forward_with_context(self, h, cand_keys_sel, cand_y_sel, sims):
        """h: (B,d); cand_keys_sel: (B,m,d); cand_y_sel: (B,m); sims: (B,m)."""
        kx = self.key(h)                                   # (B, d)
        alpha = torch.softmax(sims * self.scale, dim=1)    # (B, m)
        v = self.label_emb(cand_y_sel.unsqueeze(-1)) \
            + self.t(cand_keys_sel - kx.unsqueeze(1))      # (B, m, d)
        ctx = (alpha.unsqueeze(-1) * v).sum(dim=1)         # (B, d)
        return self.head(h + ctx).squeeze(-1)


def chunked_topm(kx: torch.Tensor, cand_keys: torch.Tensor, m: int,
                 chunk: int, exclude: torch.Tensor | None = None):
    """Top-m candidates by -squared-distance, scanning candidates in chunks.
    kx: (B,d), cand_keys: (N,d), exclude: (B,) candidate indices to mask (self).
    Returns (sims (B,m), idx (B,m))."""
    best_s, best_i = None, None
    n = len(cand_keys)
    for s in range(0, n, chunk):
        block = cand_keys[s:s + chunk]                     # (C, d)
        sims = -torch.cdist(kx, block).pow(2)              # (B, C)
        if exclude is not None:
            mask = (exclude.unsqueeze(1) >= s) & (exclude.unsqueeze(1) < s + len(block))
            rows = mask.any(dim=1)
            if rows.any():
                sims[rows, (exclude[rows] - s)] = float("-inf")
        idx = torch.arange(s, s + len(block), device=kx.device).expand(len(kx), -1)
        if best_s is None:
            best_s, best_i = sims, idx
        else:
            best_s = torch.cat([best_s, sims], dim=1)
            best_i = torch.cat([best_i, idx], dim=1)
        if best_s.shape[1] > 4 * m:                        # keep memory bounded
            top = best_s.topk(m, dim=1)
            best_i = best_i.gather(1, top.indices)
            best_s = top.values
    top = best_s.topk(min(m, best_s.shape[1]), dim=1)
    return top.values, best_i.gather(1, top.indices)


class TabRModel(TabularModel):
    name = "tabr"

    def fit(self, X, y, X_val, y_val):
        p = self.params
        device = p.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        # candidate subsample (plan: 500K max)
        n_sub = min(int(p.get("train_subsample", 500000)), len(X))
        sub = rng.choice(len(X), size=n_sub, replace=False)
        X, y = X.iloc[sub].reset_index(drop=True), np.asarray(y)[sub]

        self.prep = Preprocessor().fit(X)
        xn, xc = self.prep.transform(X, "cpu")
        y_t = torch.tensor(y.astype(np.float32))
        vxn, vxc = self.prep.transform(X_val, "cpu")
        vy = torch.tensor(np.asarray(y_val, dtype=np.float32))

        d = int(p.get("d_main", 96))
        self.m = int(p.get("context_size", 96))
        self.chunk = int(p.get("candidate_chunk", 65536))
        self.net = TabRNet(xn.shape[1], self.prep.cards, d,
                           float(p.get("dropout", 0.1))).to(device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(p.get("lr", 5e-4)),
                                weight_decay=float(p.get("weight_decay", 1e-5)))
        stopper = EarlyStopper(int(p.get("patience", 10)))
        bs = int(p.get("batch_size", 1024))
        freeze_at = int(p.get("context_freeze_epoch", 4))
        n = len(xn)
        frozen_keys = None

        for epoch in range(int(p.get("max_epochs", 100))):
            # candidate keys: recomputed each epoch until frozen
            if frozen_keys is None or epoch < freeze_at:
                with torch.no_grad():
                    keys = self._all_keys(xn, xc, device, bs)
                if epoch == freeze_at - 1 or freeze_at == 0:
                    frozen_keys = keys
            cand_keys = frozen_keys if frozen_keys is not None else keys

            self.net.train()
            perm = torch.randperm(n)
            for s in range(0, n, bs):
                b = perm[s:s + bs]
                xb_n, xb_c = xn[b].to(device), xc[b].to(device)
                yb = y_t[b].to(device)
                h = self.net.encode(xb_n, xb_c)
                kx = self.net.key(h)
                with torch.no_grad():
                    sims, idx = chunked_topm(kx.detach(), cand_keys, self.m,
                                             self.chunk, exclude=b.to(device))
                sel_keys = cand_keys[idx]                  # (B, m, d) frozen
                sel_y = y_t.to(device)[idx]
                pred = self.net.forward_with_context(h, sel_keys, sel_y, sims)
                loss = (pred - yb).abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

            # validation
            self.net.eval()
            with torch.no_grad():
                cand_eval = frozen_keys if frozen_keys is not None else \
                    self._all_keys(xn, xc, device, bs)
                val_pred = self._predict_from(vxn, vxc, cand_eval, y_t, device, bs)
                val_mae = (val_pred.cpu() - vy).abs().mean().item()
            if stopper.step(val_mae, self.net):
                break
        stopper.restore(self.net)
        # final frozen candidate state for inference
        with torch.no_grad():
            self.cand_keys = self._all_keys(xn, xc, device, bs)
        self.cand_y = y_t
        self.device, self.bs = device, bs
        return self

    def _all_keys(self, xn, xc, device, bs):
        keys = []
        for s in range(0, len(xn), bs):
            h = self.net.encode(xn[s:s + bs].to(device), xc[s:s + bs].to(device))
            keys.append(self.net.key(h))
        return torch.cat(keys)

    def _predict_from(self, xn, xc, cand_keys, cand_y, device, bs):
        out = []
        cand_y_dev = cand_y.to(device)
        for s in range(0, len(xn), bs):
            h = self.net.encode(xn[s:s + bs].to(device), xc[s:s + bs].to(device))
            kx = self.net.key(h)
            sims, idx = chunked_topm(kx, cand_keys, self.m, self.chunk)
            out.append(self.net.forward_with_context(
                h, cand_keys[idx], cand_y_dev[idx], sims))
        return torch.cat(out)

    @torch.no_grad()
    def predict(self, X) -> np.ndarray:
        self.net.eval()
        xn, xc = self.prep.transform(X, "cpu")
        return self._predict_from(xn, xc, self.cand_keys, self.cand_y,
                                  self.device, self.bs).cpu().numpy()
```

- [ ] **Step 3: Run test** — `pytest tests/test_models_torch.py -q -k tabr`

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: TabR retrieval-augmented model in PyTorch"`

---

### Task 11: Experiment runner

**Files:**
- Create: `scripts/run_experiment.py`
- Test: `tests/test_run_experiment.py` (integration on synthetic parquet)

- [ ] **Step 1: Write the failing integration test**

```python
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
```

- [ ] **Step 2: Implement `scripts/run_experiment.py`**

```python
"""Run one (model, experiment) cell of the master table.

  python scripts/run_experiment.py --model lightgbm --experiment e2
  python scripts/run_experiment.py --model tabpfn --experiment e3   # 5 seeds

Writes results/{model}_{experiment}.json immediately on completion plus
results/{model}_{experiment}_items.parquet (per-item absolute errors for
Wilcoxon tests / re-aggregation)."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from shift_study.features import FEATURES, TARGET
from shift_study.metrics import per_item_errors, wape
from shift_study.models.base import get_model
from shift_study.splits import TEST_START, e1_split, e2_split, e3_split

SPLITS = {"e1": e1_split, "e2": e2_split, "e3": e3_split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["xgboost", "lightgbm", "tabpfn", "tabm", "tabr"])
    ap.add_argument("--experiment", required=True, choices=["e1", "e2", "e3"])
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=None,
                    help="restrict to N random series (smoke test)")
    args = ap.parse_args()

    cfg_path = args.config or f"configs/{args.model}.yaml"
    config = yaml.safe_load(Path(cfg_path).read_text())
    seeds = config.get("seeds", [42])

    t0 = time.time()
    df = pd.read_parquet(args.features)
    if args.sample:
        rng = np.random.default_rng(0)
        ids = df["id"].unique()
        keep = rng.choice(ids, size=min(args.sample, len(ids)), replace=False)
        df = df[df["id"].isin(keep)]
    print(f"Loaded {len(df):,} rows, {df['id'].nunique():,} series")

    if args.experiment == "e1":
        train, val, test = e1_split(df, seed=42)
    else:
        train, val, test = SPLITS[args.experiment](df)
        assert test["d_int"].min() >= TEST_START > train["d_int"].max(), \
            "temporal leakage: test days overlap train days"
    print(f"train={len(train):,} val={len(val):,} test={len(test):,}")

    X_tr, y_tr = train[FEATURES], train[TARGET].to_numpy(dtype=np.float32)
    X_va, y_va = val[FEATURES], val[TARGET].to_numpy(dtype=np.float32)
    X_te, y_te = test[FEATURES], test[TARGET].to_numpy(dtype=np.float32)

    wapes, preds = [], []
    for seed in seeds:
        model = get_model(args.model, config, seed=seed)
        model.fit(X_tr, y_tr, X_va, y_va)
        pred = model.predict(X_te)
        w = wape(y_te, pred)
        print(f"  seed={seed}  WAPE={w:.4f}")
        wapes.append(w)
        preds.append(pred)

    mean_pred = np.mean(preds, axis=0)
    items = per_item_errors(test["id"].to_numpy(), y_te, mean_pred)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "experiment": args.experiment,
        "wape_mean": float(np.mean(wapes)),
        "wape_std": float(np.std(wapes)),
        "wapes_per_seed": [float(w) for w in wapes],
        "seeds": seeds,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "n_series_test": int(test["id"].nunique()),
        "sample": args.sample,
        "runtime_sec": round(time.time() - t0, 1),
        "config": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stem = f"{args.model}_{args.experiment}"
    (out / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    items.to_parquet(out / f"{stem}_items.parquet", index=False)
    print(f"WAPE = {payload['wape_mean']:.4f} ± {payload['wape_std']:.4f}  "
          f"-> {out / (stem + '.json')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run test** — `pytest tests/test_run_experiment.py -q`

- [ ] **Step 4: Commit** — `git add -A && git commit -m "feat: experiment runner (model x experiment -> JSON + per-item errors)"`

---

### Task 12: Crossover curve script

**Files:**
- Create: `scripts/run_crossover.py`

- [ ] **Step 1: Implement**

```python
"""Cold-start crossover curve (Figure 2): TabPFN v2 vs LightGBM WAPE as a
function of training-history size. Buckets items by number of training rows
(d < 1400) and evaluates both models on each bucket's year-5 rows.

LightGBM: one model trained on the full E2 train split (the realistic setup).
TabPFN: per-bucket context sampled from that bucket's training rows, 5 seeds."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from shift_study.features import FEATURES, TARGET
from shift_study.metrics import wape
from shift_study.models.base import get_model
from shift_study.splits import VAL_START, e2_split

BUCKET_EDGES = [10, 20, 35, 50, 75, 100, 150, 200, 300, 500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.features)
    if args.sample:
        rng = np.random.default_rng(0)
        ids = df["id"].unique()
        df = df[df["id"].isin(rng.choice(ids, min(args.sample, len(ids)), False))]

    train, val, test = e2_split(df)
    hist = train.groupby("id", observed=True).size()
    # bucket assignment: history size <= edge, smallest edge wins
    buckets = {}
    for edge_lo, edge_hi in zip([0] + BUCKET_EDGES[:-1], BUCKET_EDGES):
        ids = hist[(hist > edge_lo) & (hist <= edge_hi)].index
        if len(ids):
            buckets[edge_hi] = set(ids)
    print({k: len(v) for k, v in buckets.items()}, "items per bucket")

    # --- LightGBM: single model on full E2 train ---
    lgb_cfg = yaml.safe_load(Path("configs/lightgbm.yaml").read_text())
    lgb = get_model("lightgbm", lgb_cfg, seed=42)
    lgb.fit(train[FEATURES], train[TARGET].to_numpy(np.float32),
            val[FEATURES], val[TARGET].to_numpy(np.float32))

    pfn_cfg = yaml.safe_load(Path("configs/tabpfn.yaml").read_text())

    rows = []
    for edge, ids in sorted(buckets.items()):
        te = test[test["id"].isin(ids)]
        if te.empty:
            continue
        y_te = te[TARGET].to_numpy(np.float32)
        w_lgb = wape(y_te, lgb.predict(te[FEATURES]))

        tr_b = train[train["id"].isin(ids)]
        pfn_wapes = []
        for seed in pfn_cfg.get("seeds", [0, 1, 2, 3, 4]):
            pfn = get_model("tabpfn", pfn_cfg, seed=seed)
            pfn.fit(tr_b[FEATURES], tr_b[TARGET].to_numpy(np.float32))
            pfn_wapes.append(wape(y_te, pfn.predict(te[FEATURES])))
        rows.append({"history_max": edge, "n_items": len(ids),
                     "n_test_rows": len(te), "wape_lightgbm": w_lgb,
                     "wape_tabpfn_mean": float(np.mean(pfn_wapes)),
                     "wape_tabpfn_std": float(np.std(pfn_wapes))})
        print(rows[-1])

    res = pd.DataFrame(rows)
    crossover_n = None
    for _, r in res.iterrows():
        if r["wape_lightgbm"] < r["wape_tabpfn_mean"]:
            crossover_n = int(r["history_max"])
            break

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "crossover.csv", index=False)
    (out / "crossover.json").write_text(json.dumps({
        "crossover_n": crossover_n,
        "buckets": rows,
        "runtime_sec": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"Crossover N = {crossover_n}  -> {out / 'crossover.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test** — `python scripts/run_crossover.py --sample 300` (requires data + tabpfn; deferred to server).

- [ ] **Step 3: Commit** — `git add -A && git commit -m "feat: cold-start crossover curve script"`

---

### Task 13: Optional Optuna tuning script

**Files:**
- Create: `scripts/tune.py`

- [ ] **Step 1: Implement**

```python
"""Optional hyperparameter tuning on the E2 validation window ONLY
(train d<1400, validate 1400..1521 - the test set is never touched).
Writes the best params back into configs/{model}.yaml (backup kept as .bak).

  python scripts/tune.py --model lightgbm --trials 30 --sample 3000
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from shift_study.features import FEATURES, TARGET
from shift_study.metrics import wape
from shift_study.models.base import get_model
from shift_study.splits import e2_split


def search_space(trial, model):
    if model == "xgboost":
        return {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
            "tree_method": "hist", "objective": "reg:absoluteerror",
        }
    if model == "lightgbm":
        return {
            "objective": "regression_l1", "n_estimators": 2000, "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.9),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }
    if model == "tabm":
        return {
            "d_main": trial.suggest_categorical("d_main", [256, 384, 512]),
            "k": 8, "n_blocks": trial.suggest_int("n_blocks", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            "lr": trial.suggest_float("lr", 3e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
            "batch_size": 4096, "max_epochs": 100, "patience": 10, "device": "cuda",
        }
    if model == "tabr":
        return {
            "d_main": trial.suggest_categorical("d_main", [96, 128, 256]),
            "context_size": trial.suggest_categorical("context_size", [96, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lr": trial.suggest_float("lr", 5e-4, 1e-3, log=True),
            "weight_decay": 1e-5, "batch_size": 1024, "max_epochs": 50,
            "patience": 5, "context_freeze_epoch": 4,
            "train_subsample": 500000, "candidate_chunk": 65536, "device": "cuda",
        }
    raise SystemExit(f"No tuning for {model} (tabpfn has no hyperparameters to tune)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["xgboost", "lightgbm", "tabm", "tabr"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--sample", type=int, default=None,
                    help="tune on N series for speed")
    args = ap.parse_args()

    df = pd.read_parquet(args.features)
    if args.sample:
        rng = np.random.default_rng(0)
        ids = df["id"].unique()
        df = df[df["id"].isin(rng.choice(ids, min(args.sample, len(ids)), False))]
    train, val, _ = e2_split(df)
    X_tr, y_tr = train[FEATURES], train[TARGET].to_numpy(np.float32)
    X_va, y_va = val[FEATURES], val[TARGET].to_numpy(np.float32)

    def objective(trial):
        params = search_space(trial, args.model)
        cfg = {"params": params, "early_stopping_rounds": 75}
        m = get_model(args.model, cfg, seed=42)
        m.fit(X_tr, y_tr, X_va, y_va)
        return wape(y_va, m.predict(X_va))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    print("Best val WAPE:", study.best_value)
    print("Best params:", study.best_params)

    cfg_path = Path(f"configs/{args.model}.yaml")
    shutil.copy(cfg_path, cfg_path.with_suffix(".yaml.bak"))
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["params"].update(study.best_params)
    cfg["tuned_val_wape"] = float(study.best_value)
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"Updated {cfg_path} (backup at {cfg_path}.bak)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit** — `git add -A && git commit -m "feat: optional Optuna tuning on validation window"`

---

### Task 14: Report generator (master table, figures, Wilcoxon)

**Files:**
- Create: `scripts/make_report.py`

- [ ] **Step 1: Implement**

```python
"""Assemble the paper artifacts from results/:
  - master table (5 models x E1/E2/E3 WAPE + Delta_E2 + Delta_E3 + sensitivity)
  - Figure 1: grouped Delta bar chart
  - Figure 2: cold-start crossover curve
  - pairwise Wilcoxon signed-rank tests per experiment
  - report.md tying it together; warns if any Delta <= 0 (leakage indicator)."""
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from shift_study.metrics import delta, shift_sensitivity, wilcoxon_items

MODELS = ["xgboost", "lightgbm", "tabpfn", "tabm", "tabr"]
EXPERIMENTS = ["e1", "e2", "e3"]


def load_results(res_dir: Path) -> pd.DataFrame:
    rows = []
    for m in MODELS:
        row = {"model": m}
        for e in EXPERIMENTS:
            f = res_dir / f"{m}_{e}.json"
            if f.exists():
                p = json.loads(f.read_text())
                row[f"{e}_wape"] = p["wape_mean"]
                row[f"{e}_std"] = p["wape_std"]
        if "e1_wape" in row:
            if "e2_wape" in row:
                row["delta_e2"] = delta(row["e2_wape"], row["e1_wape"])
            if "e3_wape" in row:
                row["delta_e3"] = delta(row["e3_wape"], row["e1_wape"])
            if "delta_e2" in row and "delta_e3" in row:
                row["sensitivity"] = shift_sensitivity(row["delta_e3"], row["delta_e2"])
        rows.append(row)
    return pd.DataFrame(rows)


def fig1_deltas(table: pd.DataFrame, out: Path):
    sub = table.dropna(subset=["delta_e2", "delta_e3"], how="all")
    if sub.empty:
        return
    x = range(len(sub))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - w / 2 for i in x], sub["delta_e2"] * 100, w,
           label="Delta E2 (temporal)", color="teal")
    ax.bar([i + w / 2 for i in x], sub["delta_e3"] * 100, w,
           label="Delta E3 (cold-start)", color="mediumpurple")
    ax.set_xticks(list(x), sub["model"])
    ax.set_ylabel("WAPE degradation vs E1 (%)")
    ax.set_title("Shift degradation by model and shift type")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_shift_sensitivity.png", dpi=200)


def fig2_crossover(res_dir: Path, out: Path):
    f = res_dir / "crossover.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["history_max"], df["wape_lightgbm"], "o-", label="LightGBM")
    ax.errorbar(df["history_max"], df["wape_tabpfn_mean"],
                yerr=df["wape_tabpfn_std"], fmt="s-", label="TabPFN v2")
    cj = res_dir / "crossover.json"
    if cj.exists():
        n = json.loads(cj.read_text()).get("crossover_n")
        if n:
            ax.axvline(n, ls="--", color="gray", label=f"crossover N = {n}")
    ax.set_xscale("log")
    ax.set_xlabel("training history (rows)")
    ax.set_ylabel("WAPE")
    ax.set_title("Cold-start crossover: TabPFN v2 vs LightGBM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_crossover.png", dpi=200)


def wilcoxon_table(res_dir: Path) -> pd.DataFrame:
    rows = []
    for e in EXPERIMENTS:
        for a, b in itertools.combinations(MODELS, 2):
            fa, fb = res_dir / f"{a}_{e}_items.parquet", res_dir / f"{b}_{e}_items.parquet"
            if fa.exists() and fb.exists():
                try:
                    stat, p = wilcoxon_items(pd.read_parquet(fa), pd.read_parquet(fb))
                    rows.append({"experiment": e, "model_a": a, "model_b": b,
                                 "stat": stat, "p_value": p})
                except ValueError:
                    pass
    return pd.DataFrame(rows)


def main():
    res_dir = Path("results")
    out = Path("results/report")
    out.mkdir(parents=True, exist_ok=True)

    table = load_results(res_dir)
    table.to_csv(out / "master_table.csv", index=False)
    fig1_deltas(table, out)
    fig2_crossover(res_dir, out)
    wt = wilcoxon_table(res_dir)
    if not wt.empty:
        wt.to_csv(out / "wilcoxon.csv", index=False)

    lines = ["# Results Report", "", "## Master table", "",
             table.to_markdown(index=False, floatfmt=".4f"), ""]
    bad = table[(table.get("delta_e2", pd.Series(dtype=float)) <= 0)
                | (table.get("delta_e3", pd.Series(dtype=float)) <= 0)]
    if not bad.empty:
        lines += ["", "**WARNING: zero/negative Delta detected for: "
                  + ", ".join(bad["model"]) + " - investigate for data leakage "
                  "before reporting (see plan, Common Pitfalls).**", ""]
    if not wt.empty:
        lines += ["## Wilcoxon signed-rank (per-item WAPE)", "",
                  wt.to_markdown(index=False, floatfmt=".4g"), ""]
    lines += ["## Figures", "", "![fig1](fig1_shift_sensitivity.png)", "",
              "![fig2](fig2_crossover.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    print(table.to_string(index=False))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test** — after at least two runner outputs exist: `python scripts/make_report.py`. With no results it should still produce an (empty-ish) report without crashing.

- [ ] **Step 3: Commit** — `git add -A && git commit -m "feat: report generator (master table, figures, Wilcoxon)"`

---

### Task 15: README + final verification

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README** covering: project goal (1 paragraph), setup (`pip install -r requirements.txt && pip install -e .`, kaggle.json), the exact run order:

```bash
python scripts/download_data.py
python scripts/build_features.py                  # ~42M rows, run once
pytest -q                                         # unit tests
# smoke test the whole pipeline first:
python scripts/run_experiment.py --model lightgbm --experiment e2 --sample 500
# full study (per model x experiment):
for m in xgboost lightgbm tabpfn tabm tabr; do
  for e in e1 e2 e3; do python scripts/run_experiment.py --model $m --experiment $e; done
done
python scripts/run_crossover.py
python scripts/make_report.py
```

plus the optional tuning step, expected runtimes table from the plan, and the results-directory layout.

- [ ] **Step 2: Full test suite** — `pytest -q` → all green (model tests skip if heavy deps missing locally).

- [ ] **Step 3: Commit** — `git add -A && git commit -m "docs: README with full run order"`

---

## Self-Review Notes

- **Spec coverage:** E1/E2/E3 (Task 3, 11), features incl. leakage rule (Task 4), WAPE/Delta/sensitivity/Wilcoxon (Task 2, 14), 5 models (Tasks 7–10), crossover curve (Task 12), optional tuning (Task 13), Kaggle download (Task 5), per-item errors + immediate JSON saves (Task 11), `--sample` smoke tests (Tasks 6, 11, 12, 13). Out-of-scope items match the spec.
- **Consistency:** `FEATURES`/`TARGET` defined once in `features.py` and imported everywhere; `get_model(name, config, seed)` signature uniform; splits constants imported from `splits.py` only.
- **Env caveat:** heavy-dep tests use `pytest.importorskip`; full-data scripts are smoke-tested on the server, not in this repo's CI.





