"""
decision_engine.py - InterventionEngine

Converts calibrated vulnerability probability scores into tiered
intervention decisions, aligned with FCA Consumer Duty requirements.

Intervention tiers
------------------
  ESCALATE  - High vulnerability (P >= threshold_escalate)
              Action: Proactive welfare call within 48h, offer payment
              restructuring, refer to debt charity (StepChange).

  OUTREACH  - Moderate vulnerability (threshold_outreach <= P < threshold_escalate)
              Action: Proactive digital contact (app notification, email),
              offer budgeting tools, flag for relationship manager review.

  MONITOR   - Low vulnerability (P < threshold_outreach)
              Action: Continue standard monitoring cycle. Re-score next month.

Design note
-----------
Threshold values are NOT hardcoded. Call .calibrate_thresholds() on
the validation set after training to find the cost-minimising thresholds.
This mirrors the approach in the Stanbic pipeline but for a 3-tier output.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .constants import FN_COST, FP_COST, THRESHOLD_ESCALATE, THRESHOLD_OUTREACH

logger = logging.getLogger(__name__)

TIERS = ["MONITOR", "OUTREACH", "ESCALATE"]


@dataclass
class InterventionDecision:
    """The full output of a single vulnerability scoring event."""

    customer_id: str
    vulnerability_score: float  # Calibrated P(vulnerable)
    tier: str  # MONITOR / OUTREACH / ESCALATE
    threshold_escalate: float
    threshold_outreach: float
    top_shap_features: list = field(default_factory=list)  # From explainability module
    model_version: str = "unknown"
    scored_at: str = ""

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id,
            "vulnerability_score": round(self.vulnerability_score, 4),
            "tier": self.tier,
            "threshold_escalate": self.threshold_escalate,
            "threshold_outreach": self.threshold_outreach,
            "top_shap_features": self.top_shap_features,
            "model_version": self.model_version,
            "scored_at": self.scored_at,
        }


class InterventionEngine:
    """
    Post-model decision layer that converts probability scores to
    Consumer Duty-aligned intervention tiers.

    Parameters
    ----------
    threshold_escalate : float  P >= this → ESCALATE. Default from constants.
    threshold_outreach : float  P >= this → OUTREACH. Default from constants.
    """

    def __init__(
        self,
        threshold_escalate: float = THRESHOLD_ESCALATE,
        threshold_outreach: float = THRESHOLD_OUTREACH,
    ):
        self.threshold_escalate = threshold_escalate
        self.threshold_outreach = threshold_outreach

    # -- Calibration ----------------------------------------------------------─

    def calibrate_thresholds(
        self,
        y_val: np.ndarray,
        y_prob_val: np.ndarray,
        fn_cost: float = FN_COST,
        fp_cost: float = FP_COST,
        grid_steps: int = 100,
    ) -> dict:
        """
        Find cost-minimising thresholds on the VALIDATION SET.

        Strategy: grid search over threshold_escalate values, compute
        the business cost (fn_cost * FN + fp_cost * FP) at each, and
        select the threshold that minimises it.

        threshold_outreach is set to 60% of threshold_escalate as a
        heuristic - adjust based on intervention capacity.

        Parameters
        ----------
        y_val       : true labels (0/1 array)
        y_prob_val  : calibrated probabilities from model
        fn_cost     : cost of missing a vulnerable customer (default 5)
        fp_cost     : cost of a false alarm (default 1)
        grid_steps  : number of threshold values to try

        Returns
        -------
        dict with optimal thresholds and the cost at each
        """
        thresholds = np.linspace(0.1, 0.9, grid_steps)
        best_cost = np.inf
        best_thresh = self.threshold_escalate

        for t in thresholds:
            preds = (y_prob_val >= t).astype(int)
            fn = ((preds == 0) & (y_val == 1)).sum()
            fp = ((preds == 1) & (y_val == 0)).sum()
            cost = fn_cost * fn + fp_cost * fp
            if cost < best_cost:
                best_cost = cost
                best_thresh = t

        self.threshold_escalate = round(float(best_thresh), 4)
        self.threshold_outreach = round(float(best_thresh * 0.60), 4)

        logger.info(
            "Calibrated thresholds - ESCALATE: %.4f, OUTREACH: %.4f (cost: %.0f)",
            self.threshold_escalate,
            self.threshold_outreach,
            best_cost,
        )
        return {
            "threshold_escalate": self.threshold_escalate,
            "threshold_outreach": self.threshold_outreach,
            "min_cost": float(best_cost),
        }

    # -- Prediction ------------------------------------------------------------

    def predict_tier(self, probability: float) -> str:
        """Convert a single probability score to a tier label."""
        if probability >= self.threshold_escalate:
            return "ESCALATE"
        elif probability >= self.threshold_outreach:
            return "OUTREACH"
        return "MONITOR"

    def predict_single(
        self,
        customer_id: str,
        probability: float,
        shap_features: list = None,
        model_version: str = "unknown",
    ) -> InterventionDecision:
        """Score a single customer and return a full InterventionDecision."""
        from datetime import datetime, timezone

        return InterventionDecision(
            customer_id=customer_id,
            vulnerability_score=probability,
            tier=self.predict_tier(probability),
            threshold_escalate=self.threshold_escalate,
            threshold_outreach=self.threshold_outreach,
            top_shap_features=shap_features or [],
            model_version=model_version,
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        id_col: str = "SK_ID_CURR",
        prob_col: str = "vulnerability_score",
    ) -> pd.DataFrame:
        """
        Score a batch of customers.

        Parameters
        ----------
        df       : DataFrame with at least id_col and prob_col columns
        id_col   : column name for customer identifier
        prob_col : column name for calibrated probability scores

        Returns
        -------
        Original DataFrame with 'tier' column appended
        """
        df = df.copy()
        df["tier"] = df[prob_col].apply(self.predict_tier)

        tier_counts = df["tier"].value_counts()
        logger.info(
            "Batch scored %d customers - ESCALATE: %d, OUTREACH: %d, MONITOR: %d",
            len(df),
            tier_counts.get("ESCALATE", 0),
            tier_counts.get("OUTREACH", 0),
            tier_counts.get("MONITOR", 0),
        )
        return df

    # -- Persistence ----------------------------------------------------------─

    def save_thresholds(self, path: str = "data/external/thresholds.json"):
        """Save calibrated thresholds to disk for API loading."""
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "threshold_escalate": self.threshold_escalate,
                    "threshold_outreach": self.threshold_outreach,
                },
                f,
                indent=2,
            )
        logger.info("Thresholds saved to %s", path)

    @classmethod
    def load_thresholds(cls, path: str = "data/external/thresholds.json") -> "InterventionEngine":
        """Load a previously calibrated InterventionEngine from disk."""
        import json

        with open(path) as f:
            t = json.load(f)
        return cls(
            threshold_escalate=t["threshold_escalate"],
            threshold_outreach=t["threshold_outreach"],
        )
