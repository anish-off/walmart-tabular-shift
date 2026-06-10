"""Common model interface. Every model: fit(X, y, X_val, y_val) / predict(X).
X is a DataFrame of the shared FEATURES columns (all numeric)."""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class TabularModel(ABC):
    name: str = "base"

    def __init__(self, config: dict, seed: int = 42):
        self.config = config
        self.params = dict(config.get("params", {}))
        self.seed = seed

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            X_val: pd.DataFrame, y_val: np.ndarray) -> "TabularModel": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def get_model(name: str, config: dict, seed: int = 42) -> TabularModel:
    """Registry with lazy imports so missing heavy deps only break their model."""
    if name == "xgboost":
        from .xgb import XGBModel
        return XGBModel(config, seed)
    if name == "lightgbm":
        from .lgbm import LGBMModel
        return LGBMModel(config, seed)
    if name == "tabpfn":
        from .tabpfn import TabPFNModel
        return TabPFNModel(config, seed)
    if name == "tabm":
        from .tabm import TabMModel
        return TabMModel(config, seed)
    if name == "tabr":
        from .tabr import TabRModel
        return TabRModel(config, seed)
    raise ValueError(f"Unknown model: {name}")
