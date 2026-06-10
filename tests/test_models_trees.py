import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")

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


def test_get_model_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("nope", {})


def test_predict_before_fit_raises():
    m = get_model("lightgbm", {"params": {"verbosity": -1}})
    with pytest.raises(RuntimeError, match="fit"):
        m.predict(pd.DataFrame({"f1": [1.0]}))


def test_config_n_jobs_does_not_crash():
    X, y = synth(600)
    m = get_model("lightgbm", {"params": {"objective": "regression_l1", "n_estimators": 20,
                                          "num_leaves": 15, "verbosity": -1, "n_jobs": 2},
                               "early_stopping_rounds": 5}, seed=1)
    m.fit(X[:400], y[:400], X[400:500], y[400:500])
    assert len(m.predict(X[500:])) == 100
