"""
explainability.py - SHAP-based explanations for FinWatch.

Every vulnerability score comes with a SHAP explanation - the top features
driving that customer's score, in plain English. This is a Consumer Duty
requirement: the bank must be able to explain why a customer was flagged.

Note on SHAP for CalibratedClassifierCV
----------------------------------------
CalibratedClassifierCV wraps the base model, so we extract the underlying
estimator for SHAP computation. For LightGBM/XGBoost this gives us fast
TreeExplainer. For Logistic Regression, we use LinearExplainer.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

logger = logging.getLogger(__name__)


def get_explainer(model: CalibratedClassifierCV, X_background: pd.DataFrame):
    """
    Build a SHAP explainer for a CalibratedClassifierCV model.

    Extracts the base estimator and selects the correct SHAP explainer type.
    Uses a sample of training data as the background reference distribution.
    """
    # Extract the underlying base estimator from the calibrated wrapper
    base_estimator = model.calibrated_classifiers_[0].estimator

    model_type = type(base_estimator).__name__

    if model_type in ("LGBMClassifier", "XGBClassifier"):
        logger.info("Building TreeExplainer for %s", model_type)
        explainer = shap.TreeExplainer(base_estimator)
    else:
        logger.info("Building LinearExplainer for %s", model_type)
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.LinearExplainer(base_estimator, background)

    return explainer


def compute_shap_values(
    explainer,
    X: pd.DataFrame,
    check_additivity: bool = False,
) -> np.ndarray:
    """Compute SHAP values for a DataFrame of records."""
    shap_values = explainer.shap_values(X, check_additivity=check_additivity)

    # LightGBM returns a list [neg_class, pos_class] - take positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def top_shap_features(
    shap_values: np.ndarray,
    feature_names: list,
    n: int = 5,
) -> list:
    """
    Return the top N features driving a single prediction.

    Returns a list of dicts with feature name, SHAP value, and direction.
    Positive SHAP = increases vulnerability score.
    Negative SHAP = reduces vulnerability score.
    """
    if shap_values.ndim == 1:
        values = shap_values
    else:
        values = shap_values[0]

    indices = np.argsort(np.abs(values))[::-1][:n]

    return [
        {
            "feature": feature_names[i],
            "shap_value": round(float(values[i]), 4),
            "direction": "increases_risk" if values[i] > 0 else "reduces_risk",
        }
        for i in indices
    ]


def explain_single(
    explainer,
    record: pd.DataFrame,
    feature_names: list,
    n_features: int = 5,
) -> list:
    """
    Generate a SHAP explanation for a single customer record.

    Used at inference time in the FastAPI endpoint.
    Returns top N features in a JSON-serialisable format.
    """
    shap_vals = compute_shap_values(explainer, record)
    return top_shap_features(shap_vals, feature_names, n=n_features)
