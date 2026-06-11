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
    if wape_base == 0:
        return float("nan")
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
    """Wilcoxon signed-rank (two-sided) on per-item WAPE of model A vs model B.

    Pairs items by id, drops any item whose actual is zero in either model,
    and drops ties (wa == wb).  Returns (nan, nan) when fewer than 2 valid
    pairs remain (no id overlap, all ties, or all-zero actuals).
    """
    fa = pd.DataFrame({"id": err_a["id"], "wape": item_wape(err_a).values})
    fb = pd.DataFrame({"id": err_b["id"], "wape": item_wape(err_b).values})
    m = fa.merge(fb, on="id", suffixes=("_a", "_b"))
    wa, wb = m["wape_a"], m["wape_b"]
    mask = wa.notna() & wb.notna() & (wa != wb)
    if mask.sum() < 2:
        return (float("nan"), float("nan"))
    stat, p = wilcoxon(wa[mask], wb[mask])
    return float(stat), float(p)
