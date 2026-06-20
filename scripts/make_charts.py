"""Generate paper-quality charts from predictions_walmart.parquet.

  python scripts/make_charts.py --preds predictions_walmart.parquet --out paper_stuff/images

Produces:
  fig2_actual_vs_pred.png   -- line chart: actual vs predicted for top item
  fig3_wape_by_model.png    -- horizontal bar chart: WAPE across E1/E2/E3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 200,
})


def fig2_line_chart(df: pd.DataFrame, out: Path):
    """Actual vs predicted line chart for the top-selling item."""
    origin = pd.Timestamp("2011-01-29")
    df = df.copy()
    df["date"] = origin + pd.to_timedelta(df["d_int"] - 1, unit="D")

    # pick top selling item in test window
    top_id = df.groupby("id")["actual"].sum().idxmax()
    item = df[df["id"] == top_id].sort_values("date")

    # show 60 days for clarity
    item = item.head(60)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(item["date"], item["actual"], label="Actual", color="#2c7bb6",
            linewidth=1.5, marker="o", markersize=2.5)
    ax.plot(item["date"], item["pred"], label="TabM Predicted", color="#d7191c",
            linewidth=1.5, linestyle="--", marker="s", markersize=2.5)

    ax.set_title(f"Actual vs. Predicted Daily Sales — {top_id.replace('_evaluation','')}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Unit Sales")
    ax.legend()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(matplotlib.dates.WeekdayLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    path = out / "fig2_actual_vs_pred.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def fig3_wape_bar(out: Path):
    """Grouped horizontal bar chart of WAPE across all models and splits."""
    data = {
        "TabM":     [0.6099, 0.6374, 0.9154],
        "TabR":     [0.6398, 0.6605, 0.9234],
        "LightGBM": [0.6291, 0.6496, 0.9412],
        "XGBoost":  [0.6311, 0.6500, 0.9414],
        "TabPFN":   [0.6934, 0.7127, 1.0069],
    }
    splits = ["E1 (Random)", "E2 (Temporal)", "E3 (Cold-Start)"]
    colors = ["#4dac26", "#b8e186", "#d01c8b"]

    models = list(data.keys())
    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (split, color) in enumerate(zip(splits, colors)):
        vals = [data[m][i] for m in models]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=split, color=color,
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=90)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6,
               label="WAPE = 1.0 (naïve mean)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("WAPE (lower is better)")
    ax.set_title("Model Performance Across Evaluation Protocols")
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()

    path = out / "fig3_wape_by_model.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="predictions_walmart.parquet")
    ap.add_argument("--out", default="paper_stuff/images")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    df = pd.read_parquet(args.preds)

    fig2_line_chart(df, out)
    fig3_wape_bar(out)
    print("Done.")


if __name__ == "__main__":
    main()
