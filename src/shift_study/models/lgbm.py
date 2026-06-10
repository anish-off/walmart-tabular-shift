import lightgbm as lgb
import numpy as np

from .base import TabularModel


class LGBMModel(TabularModel):
    name = "lightgbm"

    def fit(self, X, y, X_val, y_val):
        params = dict(self.params)
        params.pop("random_state", None)   # seed always comes from self.seed
        params.setdefault("n_jobs", -1)    # overridable from config
        self.model = lgb.LGBMRegressor(**params, random_state=self.seed)
        self.model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.config.get("early_stopping_rounds", 75)),
                       lgb.log_evaluation(100)],
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Call fit() before predict()")
        return np.asarray(self.model.predict(X))
