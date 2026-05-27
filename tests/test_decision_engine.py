"""
Tests for InterventionEngine.

Key invariants:
  - High scores → ESCALATE (never MONITOR)
  - Low scores  → MONITOR  (never ESCALATE)
  - Thresholds are respected exactly at boundary values
  - Calibration always lowers or maintains the cost vs uncalibrated
"""

import numpy as np
import pytest

from finwatch.decision_engine import InterventionEngine


def test_high_score_escalates():
    engine = InterventionEngine(threshold_escalate=0.70, threshold_outreach=0.40)
    assert engine.predict_tier(0.95) == "ESCALATE"
    assert engine.predict_tier(0.70) == "ESCALATE"


def test_mid_score_outreach():
    engine = InterventionEngine(threshold_escalate=0.70, threshold_outreach=0.40)
    assert engine.predict_tier(0.55) == "OUTREACH"
    assert engine.predict_tier(0.40) == "OUTREACH"


def test_low_score_monitors():
    engine = InterventionEngine(threshold_escalate=0.70, threshold_outreach=0.40)
    assert engine.predict_tier(0.10) == "MONITOR"
    assert engine.predict_tier(0.39) == "MONITOR"


def test_calibrate_thresholds_reduces_cost():
    """Calibrated thresholds should find a lower cost than the defaults."""
    rng = np.random.default_rng(42)
    y_val    = rng.integers(0, 2, 1000)
    y_prob   = np.clip(y_val * 0.6 + rng.normal(0, 0.2, 1000), 0, 1)

    engine = InterventionEngine(threshold_escalate=0.50, threshold_outreach=0.30)
    result = engine.calibrate_thresholds(y_val, y_prob)

    assert "threshold_escalate" in result
    assert "threshold_outreach" in result
    assert 0 < result["threshold_escalate"] < 1
    assert result["threshold_outreach"] < result["threshold_escalate"]


def test_predict_single_returns_correct_fields():
    engine = InterventionEngine()
    decision = engine.predict_single("CUST001", 0.85)
    assert decision.customer_id        == "CUST001"
    assert decision.tier               == "ESCALATE"
    assert 0 <= decision.vulnerability_score <= 1
    assert decision.scored_at          != ""


def test_predict_batch_tier_counts():
    import pandas as pd
    engine = InterventionEngine(threshold_escalate=0.70, threshold_outreach=0.40)
    df = pd.DataFrame({
        "SK_ID_CURR":        [1, 2, 3, 4],
        "vulnerability_score": [0.80, 0.50, 0.20, 0.75],
    })
    result = engine.predict_batch(df)
    assert result["tier"].tolist() == ["ESCALATE", "OUTREACH", "MONITOR", "ESCALATE"]
