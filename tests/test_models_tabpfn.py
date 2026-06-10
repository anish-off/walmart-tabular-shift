import numpy as np
import pandas as pd
import pytest

tabpfn = pytest.importorskip("tabpfn")
from shift_study.metrics import wape
from shift_study.models.base import get_model


def test_tabpfn_fit_predict_small():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"f1": rng.normal(size=600), "f2": rng.normal(size=600)})
    y = 5 + 2 * X["f1"].to_numpy() + rng.normal(0, 0.2, 600)
    m = get_model("tabpfn", {"params": {"context_rows": 400, "predict_chunk": 100,
                                        "device": "cpu"}}, seed=0)
    try:
        m.fit(X[:500], y[:500], None, None)
    except Exception as e:
        # TabPFN v2 requires a one-time HuggingFace license acceptance to download
        # model weights. In CI / non-interactive environments without a cached token
        # the download fails. Skip rather than error so the suite stays green.
        if "license" in str(e).lower() or "token" in str(e).lower() or "auth" in str(e).lower():
            pytest.skip(f"TabPFN license/auth not available in this environment: {e}")
        raise
    pred = m.predict(X[500:])
    baseline = wape(y[500:], np.full(100, y[:500].mean()))
    assert wape(y[500:], pred) < baseline
