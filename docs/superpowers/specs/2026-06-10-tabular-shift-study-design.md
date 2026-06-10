# Design: tabular-shift-study

**Date:** 2026-06-10
**Source:** Full_Research_Plan-1.pdf (M5 Walmart temporal-shift benchmark)
**Target environment:** College GPU server (Linux, CUDA GPU). Data downloaded via Kaggle API.

## Goal

Benchmark 5 tabular models — XGBoost, LightGBM, TabPFN v2, TabM, TabR — on the M5
Walmart dataset under 3 split regimes, measuring how temporal distribution shift
(temporal split, cold-start) differentially affects gradient-boosted trees vs tabular
deep learning.

- **E1** — random 80/20 split (academic baseline; Delta denominator)
- **E2** — temporal split: train days 1–1521, test days 1522–1913, validation window
  days 1400–1521 (used for tuning/early stopping only)
- **E3** — cold-start subset: items with < 30 non-zero sales days in the training
  period; same temporal split as E2

**Metrics:** item-level WAPE (primary), Delta_E2 = WAPE_E2/WAPE_E1 − 1,
Delta_E3 = WAPE_E3/WAPE_E1 − 1, shift sensitivity (Delta_E3/Delta_E2), Wilcoxon
signed-rank across items. Plus the crossover curve (TabPFN v2 vs LightGBM WAPE as a
function of training-history size, 10→500 rows).

## Repo layout

```
tabular-shift-study/
├── configs/                  # one YAML per model (fixed HPs from plan ranges)
├── scripts/
│   ├── download_data.py      # Kaggle API → data/raw/
│   ├── build_features.py     # one-time: 42M-row feature table → data/processed/*.parquet
│   ├── run_experiment.py     # --model xgboost --experiment e2  (the workhorse)
│   ├── tune.py               # optional Optuna on val window (days 1400–1521)
│   ├── run_crossover.py      # E3 crossover curve: TabPFN vs LightGBM, threshold 10→500
│   └── make_report.py        # master 5×5 table, Figure 1 + 2, Wilcoxon tests
├── src/shift_study/
│   ├── data.py               # load, melt wide→long, merge calendar+prices, downcast dtypes
│   ├── features.py           # lags, rolling stats, calendar, price, label encoding
│   ├── splits.py             # E1 / E2 / E3 split logic
│   ├── metrics.py            # WAPE, Delta, shift sensitivity, Wilcoxon
│   └── models/               # one wrapper per model behind a common fit/predict interface
│       ├── base.py · xgb.py · lgbm.py · tabpfn.py · tabm.py · tabr.py
└── results/                  # results/{model}_{experiment}.json + per-item errors .parquet
```

## Feature set (identical for all 5 models)

- **Lags:** lag_7, lag_14, lag_28, lag_35 (per item×store series)
- **Rolling (computed on values up to D−1):** roll_mean_7, roll_mean_28, roll_std_28,
  roll_max_28
- **Calendar:** wday, month, week (ISO), is_weekend, event_type_1 (encoded),
  snap flag for the store's state
- **Price:** sell_price, price_change_7d, price_rank_cat, price_momentum_28
- **Categoricals:** item_id, dept_id, cat_id, store_id, state_id — label encoded

**Leakage rule:** every lag/rolling feature for day D uses only days ≤ D−7 (lags) or a
window ending at D−1 (rolls), implemented via `groupby(id).shift()` before rolling.
Rows with NaN lags (series start) are dropped from training, consistently for all models.

## Key technical decisions

1. **Feature pipeline runs once**, output saved as partitioned parquet with downcast
   dtypes (~42M rows ≈ 6–8 GB in memory). All models consume the same table.
2. **Common model interface** `fit(train, val) / predict(test)`; `run_experiment.py` is
   model-agnostic. One fixed config per model reused across all three experiments.
3. **TabPFN v2:** no training loop. Subsample 8,000 context rows; run 5 seeds; report
   mean ± std WAPE. Full-M5 prediction done in test-chunks.
4. **TabM:** PyTorch implementation per ICLR 2025 paper — shared-backbone MLP with k
   ensemble heads (BatchEnsemble-style adapters), AdamW, early stopping on validation MAE.
5. **TabR:** PyTorch implementation with differentiable k-NN retrieval over a 500K-row
   training subsample; context representations frozen after 4 epochs; batch size 1024.
6. **Per-item absolute errors saved** (parquet) for every run → Wilcoxon tests and
   hierarchy-level aggregation without re-running models.
7. **Crossover experiment:** bucket cold-start items by number of training rows
   (10→500), per-bucket WAPE for TabPFN v2 and LightGBM, report crossover N.
8. **Results discipline:** each run writes `results/{model}_{experiment}.json`
   immediately on completion (WAPE, config hash, runtime, seed info) — resumable.
9. **Tuning (optional):** `tune.py` runs Optuna per model against the validation window
   only, then updates the model's YAML config. Defaults shipped so experiments run
   without tuning.

## Hyperparameter defaults (from plan ranges)

- **XGBoost:** hist, reg:absoluteerror, n_estimators 1000 + early stop 50, lr 0.05,
  max_depth 7, subsample 0.8, colsample 0.8, min_child_weight 10
- **LightGBM:** regression_l1, num_leaves 127, lr 0.05, n_estimators 1500 + early stop
  75, min_child_samples 50, feature_fraction 0.8, bagging_fraction 0.8/freq 1, lambda_l2 1
- **TabPFN v2:** context 8000, 5 seeds
- **TabM:** d_main 512, k=8 heads, dropout 0.0, AdamW lr 1e-3, wd 1e-5, batch 4096,
  early stop patience 20 (max 200 epochs)
- **TabR:** d_main 96 (TabR-S), context_size 96, dropout 0.1, AdamW lr 5e-4, batch 1024,
  500K train subsample, context freeze after 4 epochs

## Error handling & verification

- Scripts validate inputs exist; print row counts and date ranges as sanity checks.
- E2/E3 assert `min(test day) > max(train day)`.
- `--sample N` flag (e.g. 500 items) on every script for end-to-end smoke tests.
- Negative-or-zero Delta in results flagged with a leakage warning (per plan).

## Out of scope

- Routing strategy (H4) — analysis can be done from saved per-item errors later.
- Hierarchy-level WAPE reporting beyond item level (derivable from saved errors).
- Paper writing.
