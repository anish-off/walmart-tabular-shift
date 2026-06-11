import numpy as np
import pandas as pd
import pytest
from shift_study.metrics import (
    wape, delta, shift_sensitivity, per_item_errors, item_wape, wilcoxon_items,
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


def test_item_wape_zero_actual_is_nan():
    df = per_item_errors(["a", "b"], [10, 0], [8, 0])
    result = item_wape(df).tolist()
    assert result[0] == pytest.approx(0.2)
    assert np.isnan(result[1])


def test_delta_zero_base_is_nan():
    assert np.isnan(delta(0.5, 0.0))


def test_wilcoxon_items_all_tie_returns_nan():
    err = per_item_errors(["a", "b", "c"], [10, 5, 3], [8, 4, 2])
    stat, p = wilcoxon_items(err, err.copy())
    assert np.isnan(stat) and np.isnan(p)


def test_wilcoxon_items_no_overlap_returns_nan():
    err_a = per_item_errors(["a"], [10], [8])
    err_b = per_item_errors(["b"], [10], [8])
    stat, p = wilcoxon_items(err_a, err_b)
    assert np.isnan(stat) and np.isnan(p)
