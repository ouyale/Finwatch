"""
psi_monitor.py - Population Stability Index monitoring.

PSI measures how much a feature's distribution has shifted between
the training population and the current scoring population.

  PSI < 0.10  → No significant change (green)
  PSI < 0.20  → Moderate shift, monitor closely (yellow)
  PSI >= 0.20 → Significant shift, schedule retraining (red)
  PSI >= 0.25 → Critical, retrain immediately (critical)

PSI is computed for every feature in the model, plus the score
distribution itself (score PSI is the most sensitive early warning).
"""

import logging

import numpy as np
import pandas as pd

from finwatch.constants import PSI_ALERT, PSI_RETRAIN, PSI_WARN

logger = logging.getLogger(__name__)


def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int = 10,
) -> float:
    """
    Compute the Population Stability Index between two distributions.

    Parameters
    ----------
    expected : array of values from the training/baseline population
    actual   : array of values from the current scoring population
    buckets  : number of bins (10 is industry standard for credit models)
    """
    # Create bins from expected distribution
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates at edges

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions, avoiding division by zero
    expected_pct = np.where(
        expected_counts == 0, 0.0001, expected_counts / len(expected)
    )
    actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def run_psi_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list = None,
) -> pd.DataFrame:
    """
    Compute PSI for all features and the score distribution.

    Returns a DataFrame with one row per feature:
      feature | psi | status | action
    """
    features = features or [
        c
        for c in baseline_df.columns
        if c in current_df.columns and pd.api.types.is_numeric_dtype(baseline_df[c])
    ]

    rows = []
    for feature in features:
        psi = compute_psi(
            baseline_df[feature].dropna().values,
            current_df[feature].dropna().values,
        )

        if psi >= PSI_RETRAIN:
            status, action = "CRITICAL", "Retrain immediately"
        elif psi >= PSI_ALERT:
            status, action = "ALERT", "Schedule retraining"
        elif psi >= PSI_WARN:
            status, action = "WARN", "Monitor closely"
        else:
            status, action = "OK", "No action needed"

        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "status": status,
                "action": action,
            }
        )

        if status in ("CRITICAL", "ALERT"):
            logger.warning("PSI %s - %s: %.4f", status, feature, psi)

    report = pd.DataFrame(rows).sort_values("psi", ascending=False)
    return report


def should_trigger_retrain(psi_report: pd.DataFrame) -> bool:
    """Return True if any feature has crossed the retraining threshold."""
    return (psi_report["psi"] >= PSI_ALERT).any()
