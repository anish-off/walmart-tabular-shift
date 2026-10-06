"""Condense results/ into frontend/public/data.json for the dashboard.

  python scripts/export_frontend_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

R = Path("results")
OUT = Path("frontend/public/data.json")
MODELS = ["xgboost", "lightgbm", "tabpfn", "tabm", "tabr"]
EXPS = ["e1", "e2", "e3"]
BINS = np.linspace(0, 2, 21)  # per-series WAPE, clipped at 2.0


def wape(g):
    return float(g.sum_abs_err.sum() / g.sum_actual.sum()) if g.sum_actual.sum() else None


def main():
    master = pd.read_csv(R / "report/master_table.csv")
    wil = pd.read_csv(R / "report/wilcoxon.csv")
    runs, hist, groups, items = [], [], [], {}
    for m in MODELS:
        for e in EXPS:
            meta = json.loads((R / f"{m}_{e}.json").read_text())
            df = pd.read_parquet(R / f"{m}_{e}_items.parquet")
            runs.append({
                "model": m, "exp": e, "wape": meta["wape_mean"], "std": meta["wape_std"],
                "seeds": len(meta["seeds"]), "n_train": meta["n_train"], "n_test": meta["n_test"],
                "n_series": meta["n_series_test"], "runtime_min": round(meta["runtime_sec"] / 60, 1),
            })
            w = (df.sum_abs_err / df.sum_actual.replace(0, np.nan)).dropna().clip(upper=2)
            counts, _ = np.histogram(w, bins=BINS)
            hist.append({"model": m, "exp": e, "counts": counts.tolist(), "median": float(w.median())})
            p = df.id.str.split("_")
            df["cat"] = p.str[0]
            df["state"] = p.str[3]
            for key in ("cat", "state"):
                for name, g in df.groupby(key):
                    groups.append({"model": m, "exp": e, "by": key, "name": name, "wape": wape(g), "n": len(g)})
            if e == "e3":
                items[m] = df.set_index("id")
    # E3 cold-start item table: id, actual volume, per-model WAPE
    base = items["lightgbm"][["sum_actual", "n"]].copy()
    for m in MODELS:
        d = items[m]
        base[m] = d.sum_abs_err / d.sum_actual.replace(0, np.nan)
    base = base.reset_index().rename(columns={"sum_actual": "actual"})
    cold = [{"id": r.id.replace("_evaluation", ""), "actual": float(r.actual), "n": int(r.n),
             **{m: (None if pd.isna(getattr(r, m)) else round(float(getattr(r, m)), 4)) for m in MODELS}}
            for r in base.itertuples()]
    OUT.write_text(json.dumps({
        "master": master.to_dict("records"), "wilcoxon": wil.to_dict("records"),
        "runs": runs, "hist": hist, "groups": groups, "cold": cold,
        "bins": BINS.tolist(),
    }, separators=(",", ":")))
    print(OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
