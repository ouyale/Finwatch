"""
Tests for fairness audit.

The fairness gate is a hard production gate - if it silently passes
bad models, real customers get harmed. These tests are critical.
"""

import pandas as pd
import pytest

from finwatch.fairness import disparate_impact_audit, fairness_gate


def make_fair_df():
    """Dataset where both genders have equal ESCALATE rates."""
    return pd.DataFrame({
        "CODE_GENDER": ["M"] * 50 + ["F"] * 50,
        "tier":        ["ESCALATE"] * 25 + ["MONITOR"] * 25 +
                       ["ESCALATE"] * 25 + ["MONITOR"] * 25,
    })


def make_unfair_df():
    """Dataset where women are escalated at 40% the rate of men - clear DIR fail."""
    return pd.DataFrame({
        "CODE_GENDER": ["M"] * 50 + ["F"] * 50,
        "tier":        ["ESCALATE"] * 40 + ["MONITOR"] * 10 +   # M: 80% escalated
                       ["ESCALATE"] * 16 + ["MONITOR"] * 34,    # F: 32% escalated → DIR=0.40
    })


def test_fair_dataset_passes():
    result = disparate_impact_audit(make_fair_df(), audit_cols=["CODE_GENDER"])
    assert result["all_passed"] is True
    assert result["results"]["CODE_GENDER"]["dir"] == pytest.approx(1.0, abs=0.01)


def test_unfair_dataset_fails():
    result = disparate_impact_audit(make_unfair_df(), audit_cols=["CODE_GENDER"])
    assert result["all_passed"] is False
    assert result["results"]["CODE_GENDER"]["dir"] < 0.80


def test_fairness_gate_raises_on_failure():
    result = disparate_impact_audit(make_unfair_df(), audit_cols=["CODE_GENDER"])
    with pytest.raises(ValueError, match="fairness gate"):
        fairness_gate(result, strict=True)


def test_fairness_gate_returns_false_non_strict():
    result = disparate_impact_audit(make_unfair_df(), audit_cols=["CODE_GENDER"])
    passed = fairness_gate(result, strict=False)
    assert passed is False


def test_missing_audit_column_skipped_gracefully():
    """If an audit column is not in the dataframe, it should be skipped without error."""
    df = make_fair_df()
    result = disparate_impact_audit(df, audit_cols=["CODE_GENDER", "NONEXISTENT_COL"])
    assert "NONEXISTENT_COL" not in result["results"]
    assert "CODE_GENDER" in result["results"]
