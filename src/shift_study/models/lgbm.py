import lightgbm as lgb
import numpy as np

from .base import TabularModel


class LGBMModel(TabularModel):
    name = "lightgbm"

    def fit(self, X, y, X_val, y_val):
        self.model = lgb.LGBMRegressor(**self.params, random_state=self.seed, n_jobs=-1)
        self.model.fit(
            X, y,
            eval_set=[(X_val, y_val)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(self.config.get("early_stopping_rounds", 75)),
                       lgb.log_evaluation(100)],
        )
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X))
