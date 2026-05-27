"""
fairness.py - Disparate Impact Audit for FinWatch.

Computes fairness metrics across protected demographic groups.
Model must pass the 4/5ths rule (DIR >= 0.80) on all FAIRNESS_AUDIT_COLS
before it is registered in the MLflow model registry.

This is a hard gate - not advisory. A model that fails the fairness
audit is NOT deployed, regardless of its accuracy metrics.

Regulatory basis
----------------
- FCA Consumer Duty (2023): firms must ensure good outcomes for all
  customer groups, including those with protected characteristics.
- 4/5ths rule: US EEOC standard adopted by FCA guidance. If the
  least-favoured group's positive rate < 80% of the most-favoured
  group's rate, the model has disparate impact.
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

from .constants import DIR_THRESHOLD, FAIRNESS_AUDIT_COLS

logger = logging.getLogger(__name__)


def disparate_impact_audit(
    df: pd.DataFrame,
    decision_col: str = "tier",
    positive_label: str = "ESCALATE",
    audit_cols: list = None,
    dir_threshold: float = DIR_THRESHOLD,
) -> dict:
    """
    Run a disparate impact audit across all fairness audit columns.

    For each demographic group column, computes:
    - Positive rate per group (fraction receiving ESCALATE or OUTREACH)
    - DIR  = least-favoured rate / most-favoured rate
    - DPD  = most-favoured rate - least-favoured rate
    - PASS = DIR >= dir_threshold (4/5ths rule)

    Parameters
    ----------
    df            : DataFrame with decision_col and audit columns present
    decision_col  : column containing tier labels (ESCALATE/OUTREACH/MONITOR)
    positive_label: which tier counts as 'positive' for DIR calculation.
                    Typically ESCALATE - but you can audit on OUTREACH too.
    audit_cols    : columns to audit. Defaults to FAIRNESS_AUDIT_COLS.
    dir_threshold : minimum DIR to pass. Default 0.80 (4/5ths rule).

    Returns
    -------
    dict with per-column results and an overall 'all_passed' flag.
    """
    audit_cols = audit_cols or FAIRNESS_AUDIT_COLS
    results = {}
    all_passed = True

    for col in audit_cols:
        if col not in df.columns:
            logger.warning("Fairness audit column '%s' not found - skipping.", col)
            continue

        group_rates = (
            df.groupby(col)[decision_col]
            .apply(lambda x: (x == positive_label).mean())
            .reset_index()
            .rename(columns={decision_col: "positive_rate"})
        )

        if group_rates.empty:
            continue

        max_rate = group_rates["positive_rate"].max()
        min_rate = group_rates["positive_rate"].min()
        dir_value = (min_rate / max_rate) if max_rate > 0 else 1.0
        dpd_value = max_rate - min_rate
        passed = dir_value >= dir_threshold

        if not passed:
            all_passed = False
            logger.warning(
                "FAIRNESS FAIL - %s: DIR=%.3f (threshold=%.2f). " "Groups: %s",
                col,
                dir_value,
                dir_threshold,
                group_rates.set_index(col)["positive_rate"].to_dict(),
            )
        else:
            logger.info("Fairness PASS - %s: DIR=%.3f", col, dir_value)

        results[col] = {
            "dir": round(float(dir_value), 4),
            "dpd": round(float(dpd_value), 4),
            "passed": passed,
            "max_rate": round(float(max_rate), 4),
            "min_rate": round(float(min_rate), 4),
            "group_rates": group_rates.set_index(col)["positive_rate"]
            .round(4)
            .to_dict(),
        }

    return {
        "all_passed": all_passed,
        "threshold": dir_threshold,
        "positive_label": positive_label,
        "results": results,
    }


def fairness_gate(audit_result: dict, strict: bool = True) -> bool:
    """
    Gate function: returns True only if the model passes all fairness checks.

    Used in the training pipeline to block model registration if fairness
    criteria are not met.

    Parameters
    ----------
    audit_result : output of disparate_impact_audit()
    strict       : if True, raise ValueError on failure (for CI/training pipelines)
                   if False, just log a warning and return False
    """
    passed = audit_result.get("all_passed", False)

    if not passed:
        failing = [
            col
            for col, res in audit_result.get("results", {}).items()
            if not res.get("passed", True)
        ]
        msg = (
            f"Model failed fairness gate on: {failing}. "
            f"Model will NOT be registered. Fix disparate impact before deployment."
        )
        if strict:
            raise ValueError(msg)
        logger.error(msg)

    return passed


def monthly_fairness_report(
    scored_df: pd.DataFrame,
    month_label: str,
    audit_cols: list = None,
) -> pd.DataFrame:
    """
    Generate a monthly fairness monitoring report.

    Compares current month's audit results against prior month.
    Returns a DataFrame suitable for saving to the monitoring log.
    """
    audit = disparate_impact_audit(scored_df, audit_cols=audit_cols)
    rows = []
    for col, res in audit["results"].items():
        rows.append(
            {
                "month": month_label,
                "feature": col,
                "dir": res["dir"],
                "dpd": res["dpd"],
                "passed": res["passed"],
                "max_rate": res["max_rate"],
                "min_rate": res["min_rate"],
            }
        )
    return pd.DataFrame(rows)
