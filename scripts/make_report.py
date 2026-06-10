"""Assemble the paper artifacts from results/:
  - master table (5 models x E1/E2/E3 WAPE + Delta_E2 + Delta_E3 + sensitivity)
  - Figure 1: grouped Delta bar chart
  - Figure 2: cold-start crossover curve
  - pairwise Wilcoxon signed-rank tests per experiment
  - report.md tying it together; warns if any Delta <= 0 (leakage indicator)."""
import argparse
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
    if "delta_e2" not in table.columns and "delta_e3" not in table.columns:
        return
    sub = table.dropna(subset=[c for c in ["delta_e2", "delta_e3"] if c in table.columns],
                       how="all")
    if sub.empty:
        return
    x = range(len(sub))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if "delta_e2" in sub.columns:
        ax.bar([i - w / 2 for i in x], sub["delta_e2"] * 100, w,
               label="Delta E2 (temporal)", color="teal")
    if "delta_e3" in sub.columns:
        ax.bar([i + w / 2 for i in x], sub["delta_e3"] * 100, w,
               label="Delta E3 (cold-start)", color="mediumpurple")
    ax.set_xticks(list(x), sub["model"])
    ax.set_ylabel("WAPE degradation vs E1 (%)")
    ax.set_title("Shift degradation by model and shift type")
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_shift_sensitivity.png", dpi=200)
    plt.close(fig)


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
    plt.close(fig)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res_dir = Path(args.results)
    out = res_dir / "report"
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
    for col in ("delta_e2", "delta_e3"):
        if col in table.columns:
            bad = table[table[col].notna() & (table[col] <= 0)]
            if not bad.empty:
                lines += ["", f"**WARNING: zero/negative {col} for: "
                          + ", ".join(bad["model"]) + " — investigate for data "
                          "leakage before reporting (see plan, Common Pitfalls).**", ""]
    if not wt.empty:
        lines += ["## Wilcoxon signed-rank (per-item WAPE)", "",
                  wt.to_markdown(index=False, floatfmt=".4g"), ""]
    lines += ["## Figures", "", "![fig1](fig1_shift_sensitivity.png)", "",
              "![fig2](fig2_crossover.png)", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
