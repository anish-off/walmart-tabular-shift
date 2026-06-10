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
    if args.sample is not None:
        rng = np.random.default_rng(0)
        ids = df["id"].unique()
        keep = rng.choice(ids, size=min(args.sample, len(ids)), replace=False)
        df = df[df["id"].isin(keep)]
    print(f"Loaded {len(df):,} rows, {df['id'].nunique():,} series")

    if args.experiment == "e1":
        train, val, test = e1_split(df, seed=42)
    else:
        train, val, test = SPLITS[args.experiment](df)
        if test.empty or train.empty:
            raise SystemExit(f"{args.experiment}: empty train or test split "
                             f"(train={len(train)}, test={len(test)}) — "
                             "with --sample, try more series")
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
