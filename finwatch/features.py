"""
features.py - Feature engineering for FinWatch.

All derived features are computed here - never inside the preprocessor
or the model. This makes them testable, explainable, and reusable at
inference time.

Convention: every function takes a DataFrame and returns a DataFrame
with new columns ADDED (never replaces existing ones).
"""

import numpy as np
import pandas as pd


def engineer_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive core financial ratio features from raw application fields.

    These ratios capture the relationship between credit obligation and
    financial capacity - the primary signals for vulnerability detection.
    """
    df = df.copy()

    # How much credit relative to income - high values signal overextension
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["credit_to_income_ratio"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)

    # Monthly repayment burden relative to income
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["annuity_to_income_ratio"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"].replace(
            0, np.nan
        )

    # Credit relative to actual goods value - large gap may signal cash extraction
    if {"AMT_CREDIT", "AMT_GOODS_PRICE"}.issubset(df.columns):
        df["credit_to_goods_ratio"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)

    # Aggregate external credit bureau scores (handle -1 sentinel)
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        valid = df[ext_cols].replace(-1, np.nan)
        df["ext_source_mean"] = valid.mean(axis=1)
        df["ext_source_min"] = valid.min(axis=1)
        df["ext_source_std"] = valid.std(axis=1).fillna(0)

    # Years employed as a percentage of age - employment stability signal
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        age_days = df["DAYS_BIRTH"].abs().replace(0, np.nan)
        employed_days = df["DAYS_EMPLOYED"].abs()
        df["days_employed_pct_of_age"] = (employed_days / age_days).clip(0, 1)

    return df


def engineer_document_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate document submission flags into a single completeness score.

    Incomplete documentation is a proxy for disorganisation or financial
    stress. The raw dataset has ~21 FLAG_DOCUMENT_* binary columns.
    """
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        df["document_completeness_score"] = df[doc_cols].sum(axis=1) / len(doc_cols)
    return df


def engineer_enquiry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise credit bureau enquiry counts across time windows.

    Multiple recent enquiries signal financial stress - person shopping
    for credit across many lenders, often because they are being declined.
    """
    enq_cols = {
        "AMT_REQ_CREDIT_BUREAU_HOUR": "enq_hour",
        "AMT_REQ_CREDIT_BUREAU_DAY": "enq_day",
        "AMT_REQ_CREDIT_BUREAU_WEEK": "enq_week",
        "AMT_REQ_CREDIT_BUREAU_MON": "enq_month",
        "AMT_REQ_CREDIT_BUREAU_QRT": "enq_quarter",
        "AMT_REQ_CREDIT_BUREAU_YEAR": "enq_year",
    }
    for raw, alias in enq_cols.items():
        if raw in df.columns:
            df[alias] = df[raw].fillna(0)

    enq_recent = [
        v
        for k, v in enq_cols.items()
        if k in df.columns and ("HOUR" in k or "DAY" in k or "WEEK" in k or "MON" in k)
    ]
    if enq_recent:
        df["total_recent_enquiries"] = df[enq_recent].sum(axis=1)

    return df


def add_macro_features(df: pd.DataFrame, macro_snapshot: dict) -> pd.DataFrame:
    """
    Append current ONS macroeconomic indicator values to every row.

    This is called at inference time with the latest ONS API snapshot,
    and at training time with the macro values current at the training date.

    Parameters
    ----------
    df             : DataFrame of customer features
    macro_snapshot : dict from finwatch.macro_data.get_macro_snapshot()
                     e.g. {"cpi_inflation_rate": 3.4, "unemployment_rate": 4.2, ...}
    """
    df = df.copy()
    for key, value in macro_snapshot.items():
        df[key] = value
    return df


def run_all(df: pd.DataFrame, macro_snapshot: dict = None) -> pd.DataFrame:
    """
    Apply all feature engineering steps in the correct order.

    Parameters
    ----------
    df             : Preprocessed (cleaned) DataFrame
    macro_snapshot : ONS macro values to append. If None, macro features
                     are skipped (use during initial EDA / offline work).
    """
    df = engineer_financial_ratios(df)
    df = engineer_document_flags(df)
    df = engineer_enquiry_features(df)

    if macro_snapshot:
        df = add_macro_features(df, macro_snapshot)

    return df
