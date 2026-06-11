"""TabPFN v2: in-context learning, no training loop. fit() samples a context
window (default 8,000 rows) with the run's seed; different seeds give different
contexts -> run 5 seeds and report mean/std (handled by run_experiment)."""
import numpy as np

from .base import TabularModel


class TabPFNModel(TabularModel):
    name = "tabpfn"

    def fit(self, X, y, X_val, y_val):  # X_val/y_val unused; TabPFN is in-context
        from tabpfn import TabPFNRegressor

        n_ctx = int(self.params.get("context_rows", 8000))
        rng = np.random.default_rng(self.seed)
        if len(X) > n_ctx:
            idx = rng.choice(len(X), size=n_ctx, replace=False)
            X_ctx, y_ctx = X.iloc[idx], np.asarray(y)[idx]
        else:
            X_ctx, y_ctx = X, np.asarray(y)
        self.model = TabPFNRegressor(
            device=self.params.get("device", "cuda"),
            random_state=self.seed,
            ignore_pretraining_limits=True,
        )
        self.model.fit(X_ctx, y_ctx)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        if len(X) == 0:
            return np.empty(0, dtype=np.float64)
        chunk = int(self.params.get("predict_chunk", 50000))
        out = []
        for start in range(0, len(X), chunk):
            out.append(np.asarray(self.model.predict(X.iloc[start:start + chunk])))
        return np.concatenate(out)
