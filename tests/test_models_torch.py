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
