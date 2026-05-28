"""
Tests for feature engineering functions.

Feature engineering is pure data transformation - no external dependencies,
no randomness, no network calls. Every function takes a DataFrame in and
returns a DataFrame out, which makes them straightforward to test.
"""

import numpy as np
import pandas as pd
import pytest

from finwatch.features import (
    add_macro_features,
    engineer_document_flags,
    engineer_enquiry_features,
    engineer_financial_ratios,
    run_all,
)


def make_base_df(n=10) -> pd.DataFrame:
    """Minimal DataFrame with the raw columns feature engineering expects."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": rng.uniform(50000, 300000, n),
            "AMT_CREDIT": rng.uniform(100000, 800000, n),
            "AMT_ANNUITY": rng.uniform(10000, 50000, n),
            "AMT_GOODS_PRICE": rng.uniform(80000, 500000, n),
            "EXT_SOURCE_1": rng.uniform(0, 1, n),
            "EXT_SOURCE_2": rng.uniform(0, 1, n),
            "EXT_SOURCE_3": rng.uniform(0, 1, n),
            "DAYS_BIRTH": rng.integers(-25000, -6000, n),
            "DAYS_EMPLOYED": rng.integers(-5000, -1, n),
            "FLAG_DOCUMENT_3": rng.integers(0, 2, n),
            "FLAG_DOCUMENT_5": rng.integers(0, 2, n),
            "FLAG_DOCUMENT_6": rng.integers(0, 2, n),
            "AMT_REQ_CREDIT_BUREAU_HOUR": rng.integers(0, 3, n),
            "AMT_REQ_CREDIT_BUREAU_DAY": rng.integers(0, 3, n),
            "AMT_REQ_CREDIT_BUREAU_WEEK": rng.integers(0, 5, n),
            "AMT_REQ_CREDIT_BUREAU_MON": rng.integers(0, 5, n),
            "AMT_REQ_CREDIT_BUREAU_QRT": rng.integers(0, 10, n),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 20, n),
        }
    )


# -- Financial ratios ---------------------------------------------------------


def test_credit_to_income_ratio_created():
    df = make_base_df()
    out = engineer_financial_ratios(df)
    assert "credit_to_income_ratio" in out.columns


def test_annuity_to_income_ratio_created():
    df = make_base_df()
    out = engineer_financial_ratios(df)
    assert "annuity_to_income_ratio" in out.columns


def test_credit_to_goods_ratio_created():
    df = make_base_df()
    out = engineer_financial_ratios(df)
    assert "credit_to_goods_ratio" in out.columns


def test_ext_source_mean_created():
    df = make_base_df()
    out = engineer_financial_ratios(df)
    assert "ext_source_mean" in out.columns


def test_days_employed_pct_clipped_between_0_and_1():
    """Employment percentage of age must stay in [0, 1]."""
    df = make_base_df()
    out = engineer_financial_ratios(df)
    assert "days_employed_pct_of_age" in out.columns
    assert (out["days_employed_pct_of_age"] >= 0).all()
    assert (out["days_employed_pct_of_age"] <= 1).all()


def test_ratio_numerics_are_correct():
    """Verify credit_to_income_ratio is computed correctly for a known row."""
    df = pd.DataFrame({"AMT_CREDIT": [200000.0], "AMT_INCOME_TOTAL": [100000.0]})
    out = engineer_financial_ratios(df)
    assert out["credit_to_income_ratio"].iloc[0] == pytest.approx(2.0)


def test_zero_income_does_not_cause_division_error():
    """Zero income must be replaced with NaN before dividing - not crash."""
    df = pd.DataFrame(
        {
            "AMT_CREDIT": [100000.0],
            "AMT_INCOME_TOTAL": [0.0],
            "AMT_ANNUITY": [5000.0],
        }
    )
    out = engineer_financial_ratios(df)
    # Result should be NaN (not inf or crash)
    assert pd.isna(out["credit_to_income_ratio"].iloc[0])


def test_missing_columns_skipped_gracefully():
    """If AMT_CREDIT is missing, ratio columns should simply not be created."""
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [100000.0]})
    out = engineer_financial_ratios(df)
    assert "credit_to_income_ratio" not in out.columns


def test_original_dataframe_not_mutated():
    """engineer_financial_ratios must not modify the input DataFrame."""
    df = make_base_df()
    original_cols = set(df.columns)
    engineer_financial_ratios(df)
    assert set(df.columns) == original_cols


# -- Document flags -----------------------------------------------------------


def test_document_completeness_score_created():
    df = make_base_df()
    out = engineer_document_flags(df)
    assert "document_completeness_score" in out.columns


def test_document_completeness_score_between_0_and_1():
    df = make_base_df()
    out = engineer_document_flags(df)
    assert (out["document_completeness_score"] >= 0).all()
    assert (out["document_completeness_score"] <= 1).all()


def test_all_docs_submitted_gives_score_1():
    df = pd.DataFrame({"FLAG_DOCUMENT_3": [1], "FLAG_DOCUMENT_5": [1]})
    out = engineer_document_flags(df)
    assert out["document_completeness_score"].iloc[0] == pytest.approx(1.0)


def test_no_doc_columns_no_score_column():
    """If no FLAG_DOCUMENT_* columns exist, no score column should be added."""
    df = pd.DataFrame({"AMT_INCOME_TOTAL": [100000.0]})
    out = engineer_document_flags(df)
    assert "document_completeness_score" not in out.columns


# -- Enquiry features ---------------------------------------------------------


def test_enquiry_aliases_created():
    df = make_base_df()
    out = engineer_enquiry_features(df)
    assert "enq_hour" in out.columns
    assert "enq_month" in out.columns
    assert "enq_year" in out.columns


def test_total_recent_enquiries_created():
    df = make_base_df()
    out = engineer_enquiry_features(df)
    assert "total_recent_enquiries" in out.columns


def test_enquiry_nulls_filled_with_zero():
    """Missing enquiry values should become 0 (no enquiry, not unknown)."""
    df = pd.DataFrame({"AMT_REQ_CREDIT_BUREAU_HOUR": [np.nan], "AMT_REQ_CREDIT_BUREAU_DAY": [2.0]})
    out = engineer_enquiry_features(df)
    assert out["enq_hour"].iloc[0] == 0.0


# -- Macro features -----------------------------------------------------------


def test_macro_features_appended_to_every_row():
    df = make_base_df(5)
    macro = {"cpi_inflation_rate": 3.4, "unemployment_rate": 4.2}
    out = add_macro_features(df, macro)
    assert "cpi_inflation_rate" in out.columns
    assert (out["cpi_inflation_rate"] == 3.4).all()
    assert (out["unemployment_rate"] == 4.2).all()


def test_macro_features_does_not_mutate_input():
    df = make_base_df(3)
    original_cols = set(df.columns)
    add_macro_features(df, {"cpi": 3.0})
    assert set(df.columns) == original_cols


# -- run_all ------------------------------------------------------------------


def test_run_all_returns_dataframe():
    df = make_base_df()
    out = run_all(df)
    assert isinstance(out, pd.DataFrame)


def test_run_all_adds_expected_columns():
    df = make_base_df()
    out = run_all(df)
    assert "credit_to_income_ratio" in out.columns
    assert "document_completeness_score" in out.columns
    assert "enq_year" in out.columns


def test_run_all_with_macro_snapshot():
    df = make_base_df()
    macro = {"cpi_inflation_rate": 3.4, "unemployment_rate": 4.2}
    out = run_all(df, macro_snapshot=macro)
    assert "cpi_inflation_rate" in out.columns


def test_run_all_without_macro_snapshot():
    """Passing no macro snapshot should not crash - just skip macro features."""
    df = make_base_df()
    out = run_all(df, macro_snapshot=None)
    assert "cpi_inflation_rate" not in out.columns
