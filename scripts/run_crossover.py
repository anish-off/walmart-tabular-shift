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
from shift_study.splits import e2_split

BUCKET_EDGES = [10, 20, 35, 50, 75, 100, 150, 200, 300, 500]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--sample", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    df = pd.read_parquet(args.features)
    if args.sample is not None:
        rng = np.random.default_rng(0)
        ids = df["id"].unique()
        df = df[df["id"].isin(rng.choice(ids, min(args.sample, len(ids)), replace=False))]

    train, val, test = e2_split(df)
    hist = train.groupby("id", observed=True).size()
    buckets = {}
    for edge_lo, edge_hi in zip([0] + BUCKET_EDGES[:-1], BUCKET_EDGES):
        ids = hist[(hist > edge_lo) & (hist <= edge_hi)].index
        if len(ids):
            buckets[edge_hi] = set(ids)
    print({k: len(v) for k, v in buckets.items()}, "items per bucket")
    if not buckets:
        raise SystemExit("No items with <= 500 training rows found; nothing to do.")

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
        va_b = val[val["id"].isin(ids)]
        pfn_wapes = []
        for seed in pfn_cfg.get("seeds", [0, 1, 2, 3, 4]):
            pfn = get_model("tabpfn", pfn_cfg, seed=seed)
            pfn.fit(tr_b[FEATURES], tr_b[TARGET].to_numpy(np.float32),
                    va_b[FEATURES], va_b[TARGET].to_numpy(np.float32))
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
