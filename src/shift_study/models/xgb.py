import numpy as np
import xgboost as xgb

from .base import TabularModel


class XGBModel(TabularModel):
    name = "xgboost"

    def fit(self, X, y, X_val, y_val):
        # XGBoost >= 2.1 removed early_stopping_rounds from fit(); use callback instead.
        early_stopping_rounds = self.config.get("early_stopping_rounds", 50)
        callbacks = [xgb.callback.EarlyStopping(rounds=early_stopping_rounds)]
        params = dict(self.params)
        params.pop("random_state", None)   # seed always comes from self.seed
        params.setdefault("n_jobs", -1)    # overridable from config
        self.model = xgb.XGBRegressor(
            **params,
            random_state=self.seed,
            callbacks=callbacks,
        )
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=100)
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        return np.asarray(self.model.predict(X))
