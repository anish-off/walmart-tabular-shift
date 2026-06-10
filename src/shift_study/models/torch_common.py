"""Shared helpers for the PyTorch models: column typing, standardization,
embedding sizing, tensor conversion, early stopping."""
import copy

import numpy as np
import pandas as pd
import torch

# Categorical-encoded columns (integer codes from features.py). Everything else
# in X is treated as numeric and standardized.
CAT_ENC_COLS = ["item_id_enc", "dept_id_enc", "cat_id_enc", "store_id_enc",
                "state_id_enc", "event_type_1_enc"]


def split_columns(X: pd.DataFrame):
    cat_cols = [c for c in CAT_ENC_COLS if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def emb_dim(cardinality: int) -> int:
    return int(min(64, max(4, round(1.6 * cardinality ** 0.56))))


class Preprocessor:
    """Standardize numerics (train stats); pass through cat codes with
    cardinalities; unseen/val codes clipped into range."""

    def fit(self, X: pd.DataFrame):
        self.num_cols, self.cat_cols = split_columns(X)
        num = X[self.num_cols].to_numpy(dtype=np.float32)
        self.mean = num.mean(axis=0)
        self.std = num.std(axis=0) + 1e-6
        self.cards = [int(X[c].max()) + 2 for c in self.cat_cols]  # +1 safety slot
        return self

    def transform(self, X: pd.DataFrame, device: str):
        num = (X[self.num_cols].to_numpy(dtype=np.float32) - self.mean) / self.std
        x_num = torch.tensor(num, device=device)
        if self.cat_cols:
            cat = X[self.cat_cols].to_numpy(dtype=np.int64)
            cat = np.clip(cat, 0, np.array(self.cards) - 1)
            x_cat = torch.tensor(cat, device=device)
        else:
            x_cat = torch.zeros((len(X), 0), dtype=torch.long, device=device)
        return x_num, x_cat


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.best_state = None
        self.bad_epochs = 0

    def step(self, metric: float, model: torch.nn.Module) -> bool:
        """Returns True when training should stop."""
        if metric < self.best:
            self.best = metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model: torch.nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
