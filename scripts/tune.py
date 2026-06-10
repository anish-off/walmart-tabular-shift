"""Optional hyperparameter tuning on the E2 validation window ONLY
(train d<1400, validate 1400..1521 - the test set is never touched).
Writes the best params back into configs/{model}.yaml (backup kept as .bak).

  python scripts/tune.py --model lightgbm --trials 30 --sample 3000

Notes
-----
The Optuna objective is evaluated on the *same* validation window used for
early stopping, so the reported val WAPE is optimistically biased.  This is
intentional: the bias is consistent across trials and makes tuning a fair
ranking problem.  The tuned params should still be validated on held-out test
data in the final run_experiment step.

The ``n_estimators`` / ``max_epochs`` values in ``search_space`` are fixed
tuning budgets (deliberately capped for trial speed) and differ from the
values set in the final-run configs/*.yaml files.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from shift_study.features import FEATURES, TARGET
from shift_study.metrics import wape
from shift_study.models.base import get_model
from shift_study.splits import e2_split


def search_space(trial, model):
    if model == "xgboost":
        return {
            "n_estimators": 1500,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "max_depth": trial.suggest_int("max_depth", 5, 9),
            "subsample": trial.suggest_float("subsample", 0.7, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
            "tree_method": "hist", "objective": "reg:absoluteerror",
        }
    if model == "lightgbm":
        return {
            "objective": "regression_l1", "n_estimators": 2000, "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 63, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.9),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }
    if model == "tabm":
        return {
            "d_main": trial.suggest_categorical("d_main", [256, 384, 512]),
            "k": 8, "n_blocks": trial.suggest_int("n_blocks", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.0, 0.2),
            "lr": trial.suggest_float("lr", 3e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True),
            "batch_size": 4096, "max_epochs": 100, "patience": 10, "device": "cuda",
        }
    if model == "tabr":
        return {
            "d_main": trial.suggest_categorical("d_main", [96, 128, 256]),
            "context_size": trial.suggest_categorical("context_size", [96, 128, 256]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "lr": trial.suggest_float("lr", 5e-4, 1e-3, log=True),
            "weight_decay": 1e-5, "batch_size": 1024, "max_epochs": 50,
            "patience": 5, "context_freeze_epoch": 4,
            "train_subsample": 500000, "candidate_chunk": 65536, "device": "cuda",
        }
    raise SystemExit(f"No tuning for {model} (tabpfn has no hyperparameters to tune)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["xgboost", "lightgbm", "tabm", "tabr"])
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--sample", type=int, default=None,
                    help="tune on N series for speed")
    args = ap.parse_args()

    df = pd.read_parquet(args.features, columns=["id", "d_int", "sales"] + FEATURES)
    if args.sample is not None:
        rng = np.random.default_rng(0)
        ids = df["id"].unique()
        df = df[df["id"].isin(rng.choice(ids, min(args.sample, len(ids)), replace=False))]
    train, val, _ = e2_split(df)
    X_tr, y_tr = train[FEATURES], train[TARGET].to_numpy(np.float32)
    X_va, y_va = val[FEATURES], val[TARGET].to_numpy(np.float32)

    def objective(trial):
        params = search_space(trial, args.model)
        cfg = {"params": params, "early_stopping_rounds": 75}
        m = get_model(args.model, cfg, seed=42)
        m.fit(X_tr, y_tr, X_va, y_va)
        return wape(y_va, m.predict(X_va))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.trials)
    print("Best val WAPE:", study.best_value)
    print("Best params:", study.best_params)

    cfg_path = Path(f"configs/{args.model}.yaml")
    shutil.copy(cfg_path, cfg_path.with_name(cfg_path.name + ".bak"))
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["params"].update(study.best_params)
    cfg["tuned_val_wape"] = float(study.best_value)
    cfg_path.write_text(yaml.dump(cfg, sort_keys=False))
    print(f"Updated {cfg_path} (backup at {cfg_path}.bak)")


if __name__ == "__main__":
    main()
