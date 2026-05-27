"""
models.py - Model training, calibration, and champion selection.

Trains three models in every run:
  1. Logistic Regression   - interpretable baseline, sets the floor
  2. LightGBM              - expected champion (handles missingness natively)
  3. XGBoost               - challenger (independent gradient boosting comparison)

All models are wrapped in CalibratedClassifierCV (Platt scaling) so their
output is an honest probability, not a raw score.

Champion selection metric: PR-AUC (Average Precision).
On imbalanced vulnerability data, PR-AUC is the correct metric -
accuracy and ROC-AUC reward predicting the majority class.
"""

import logging
from typing import Dict, Tuple

import mlflow
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# -- Evaluation ----------------------------------------------------------------

def evaluate_model(model, X: pd.DataFrame, y: np.ndarray, split_name: str = "val") -> dict:
    """Compute the full evaluation suite for a fitted model."""
    y_prob = model.predict_proba(X)[:, 1]
    metrics = {
        f"{split_name}_pr_auc":   round(average_precision_score(y, y_prob), 4),
        f"{split_name}_roc_auc":  round(roc_auc_score(y, y_prob), 4),
        f"{split_name}_brier":    round(brier_score_loss(y, y_prob), 4),
        f"{split_name}_gini":     round(2 * roc_auc_score(y, y_prob) - 1, 4),
    }
    # KS statistic - max separation between default and non-default distributions
    from scipy.stats import ks_2samp
    pos_scores = y_prob[y == 1]
    neg_scores = y_prob[y == 0]
    ks_stat, _ = ks_2samp(pos_scores, neg_scores)
    metrics[f"{split_name}_ks"] = round(ks_stat, 4)

    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)
    return metrics


# -- Baseline: Logistic Regression --------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> CalibratedClassifierCV:
    """Train a calibrated logistic regression baseline."""
    logger.info("Training Logistic Regression baseline...")
    base = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs",
    )
    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    model.fit(X_train, y_train)
    return model


# -- Champion: LightGBM with Optuna --------------------------------------------

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> CalibratedClassifierCV:
    """Train LightGBM with Optuna hyperparameter optimisation."""
    logger.info("Tuning LightGBM with Optuna (%d trials)...", n_trials)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight":  scale_pos_weight,
            "random_state":      42,
            "verbose":           -1,
        }
        m = LGBMClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        y_prob = m.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["random_state"] = 42
    best_params["verbose"] = -1

    logger.info("Best LightGBM PR-AUC (val): %.4f", study.best_value)
    logger.info("Best params: %s", best_params)

    base = LGBMClassifier(**best_params)
    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    model.fit(X_train, y_train)
    return model


# -- Challenger: XGBoost with Optuna ------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 50,
) -> CalibratedClassifierCV:
    """Train XGBoost with Optuna hyperparameter optimisation."""
    logger.info("Tuning XGBoost with Optuna (%d trials)...", n_trials)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight":  scale_pos_weight,
            "eval_metric":       "aucpr",
            "random_state":      42,
            "verbosity":         0,
        }
        m = XGBClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = m.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    best_params.update({"scale_pos_weight": scale_pos_weight,
                        "random_state": 42, "verbosity": 0})

    logger.info("Best XGBoost PR-AUC (val): %.4f", study.best_value)

    base = XGBClassifier(**best_params)
    model = CalibratedClassifierCV(base, cv=5, method="sigmoid")
    model.fit(X_train, y_train)
    return model


# -- Champion Selection --------------------------------------------------------

def select_champion(
    models: Dict[str, CalibratedClassifierCV],
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[str, CalibratedClassifierCV, dict]:
    """
    Select the champion model based on PR-AUC on the validation set.

    Returns (champion_name, champion_model, all_metrics)
    """
    all_metrics = {}
    best_name   = None
    best_pr_auc = -1.0

    for name, model in models.items():
        logger.info("Evaluating %s...", name)
        metrics = evaluate_model(model, X_val, y_val, split_name="val")
        all_metrics[name] = metrics
        if metrics["val_pr_auc"] > best_pr_auc:
            best_pr_auc = metrics["val_pr_auc"]
            best_name   = name

    logger.info("Champion: %s (PR-AUC=%.4f)", best_name, best_pr_auc)
    return best_name, models[best_name], all_metrics
