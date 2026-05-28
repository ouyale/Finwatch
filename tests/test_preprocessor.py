"""
Tests for CustomerPreprocessor.

Industry rule: the preprocessor is the most critical component to test
because silent bugs here (wrong column order, leakage not dropped,
scaler refit at inference) corrupt every downstream result.
"""

import numpy as np
import pandas as pd
import pytest

from finwatch.constants import DROP_COLS, PROTECTED_COLS
from finwatch.preprocessor import CustomerPreprocessor


def make_sample_df(n=100) -> pd.DataFrame:
    """Create a minimal realistic DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "SK_ID_CURR": rng.integers(100000, 999999, n),
            "AMT_INCOME_TOTAL": rng.uniform(50000, 300000, n),
            "AMT_CREDIT": rng.uniform(100000, 800000, n),
            "AMT_ANNUITY": rng.uniform(10000, 50000, n),
            "AMT_GOODS_PRICE": rng.uniform(80000, 500000, n),
            "CODE_GENDER": rng.choice(["M", "F"], n),
            "DAYS_BIRTH": rng.integers(-25000, -6000, n),
            "DAYS_EMPLOYED": rng.integers(-5000, 0, n),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "State servant"], n),
            "EXT_SOURCE_1": np.where(rng.random(n) > 0.4, rng.uniform(0, 1, n), np.nan),
            "EXT_SOURCE_2": np.where(rng.random(n) > 0.3, rng.uniform(0, 1, n), np.nan),
            "EXT_SOURCE_3": np.where(rng.random(n) > 0.6, rng.uniform(0, 1, n), np.nan),
            "REGION": rng.choice(["London", "North", "South"], n),  # protected col
        }
    )


def make_labels(n=100) -> np.ndarray:
    return np.random.default_rng(42).integers(0, 2, n)


# -- Critical: protected columns must be dropped FIRST ------------------------─


def test_protected_columns_dropped():
    """Protected and leakage columns must not appear in output."""
    df = make_sample_df()
    y = make_labels()
    pp = CustomerPreprocessor()
    out = pp.fit_transform(df, y)

    for col in PROTECTED_COLS + DROP_COLS:
        assert col not in out.columns, f"Column '{col}' should have been dropped but wasn't"


# -- Fit/transform symmetry ----------------------------------------------------─


def test_transform_matches_fit_transform_schema():
    """transform() must produce identical column schema to fit_transform()."""
    df = make_sample_df(200)
    y = make_labels(200)
    pp = CustomerPreprocessor()

    train_out = pp.fit_transform(df[:150], y[:150])
    val_out = pp.transform(df[150:])

    assert list(train_out.columns) == list(
        val_out.columns
    ), "Column order/schema mismatch between fit_transform and transform"


# -- No NaN after preprocessing ------------------------------------------------─


def test_no_nulls_after_transform():
    """Output must contain no NaN values."""
    df = make_sample_df()
    y = make_labels()
    pp = CustomerPreprocessor()
    out = pp.fit_transform(df, y)
    assert not out.isnull().any().any(), "NaN values found after preprocessing"


# -- transform_single matches batch transform ----------------------------------─


def test_transform_single_matches_batch():
    """Single-record inference must produce the same result as batch transform."""
    df = make_sample_df(50)
    y = make_labels(50)
    pp = CustomerPreprocessor()
    pp.fit(df, y)

    record = df.iloc[0].to_dict()
    single_out = pp.transform_single(record)
    batch_out = pp.transform(df.iloc[[0]])

    pd.testing.assert_frame_equal(single_out, batch_out)


# -- DAYS_EMPLOYED sentinel handling ------------------------------------------─


def test_days_employed_sentinel_handled():
    """DAYS_EMPLOYED=365243 is a sentinel for 'unemployed' and must be flagged."""
    df = make_sample_df(10)
    df.loc[0, "DAYS_EMPLOYED"] = 365243
    y = make_labels(10)
    pp = CustomerPreprocessor()
    out = pp.fit_transform(df, y)

    assert "DAYS_EMPLOYED_IS_NA" in out.columns, "Sentinel flag column missing"
    assert out.loc[0, "DAYS_EMPLOYED_IS_NA"] == 1


# -- EXT_SOURCE missing value handling ----------------------------------------─


def test_ext_source_missing_creates_flag():
    """Missing EXT_SOURCE values should create a has_ext_source_n binary flag."""
    df = make_sample_df(10)
    df["EXT_SOURCE_1"] = np.nan  # force all missing
    y = make_labels(10)
    pp = CustomerPreprocessor()
    out = pp.fit_transform(df, y)

    assert "has_ext_source_1" in out.columns
    assert (out["has_ext_source_1"] == 0).all()


# -- Unfitted error ------------------------------------------------------------─


def test_transform_without_fit_raises():
    """Calling transform() before fit() must raise a RuntimeError."""
    df = make_sample_df(10)
    pp = CustomerPreprocessor()
    with pytest.raises(RuntimeError, match="not been fitted"):
        pp.transform(df)


# -- Deduplication on SK_ID_CURR -----------------------------------------------─


def test_duplicates_removed_on_sk_id_curr():
    """Duplicate SK_ID_CURR rows must be removed before any other step."""
    df = make_sample_df(10)
    # Inject two rows with the same SK_ID_CURR - the first should be dropped
    df.loc[0, "SK_ID_CURR"] = 999999
    df.loc[1, "SK_ID_CURR"] = 999999
    y = make_labels(10)
    pp = CustomerPreprocessor()
    # fit should not raise and the output should have one fewer row
    out = pp.fit_transform(df, y)
    # After deduplication we lose one row: 10 - 1 = 9
    assert len(out) == 9


def test_no_sk_id_curr_column_does_not_crash():
    """If SK_ID_CURR is absent the deduplication step must be a silent no-op."""
    df = make_sample_df(10).drop(columns=["SK_ID_CURR"])
    y = make_labels(10)
    pp = CustomerPreprocessor()
    # Should complete without raising
    out = pp.fit_transform(df, y)
    assert len(out) == 10


# -- OWN_CAR_AGE conditional imputation ----------------------------------------─


def make_car_df() -> pd.DataFrame:
    """DataFrame with car ownership columns for testing OWN_CAR_AGE logic."""
    return pd.DataFrame(
        {
            "FLAG_OWN_CAR": [1, 1, 1, 0, 0],
            "OWN_CAR_AGE": [5.0, 10.0, np.nan, np.nan, np.nan],
            "AMT_INCOME_TOTAL": [100000.0] * 5,
        }
    )


def test_own_car_age_zero_for_non_car_owners():
    """Customers with FLAG_OWN_CAR=0 must get OWN_CAR_AGE=0, not a median."""
    df = make_car_df()
    y = np.array([0, 0, 0, 0, 0])
    pp = CustomerPreprocessor(scale=False)
    out = pp.fit_transform(df, y)
    assert "OWN_CAR_AGE" in out.columns
    non_owner_rows = out.index[df["FLAG_OWN_CAR"] == 0]
    assert (out.loc[non_owner_rows, "OWN_CAR_AGE"] == 0).all()


def test_own_car_age_uses_car_owner_median_for_car_owners_with_missing():
    """Car owners with missing OWN_CAR_AGE should receive the car-owner median."""
    df = make_car_df()
    # Car owners: ages 5 and 10 are known; the third has NaN - median = 7.5
    y = np.array([0, 0, 0, 0, 0])
    pp = CustomerPreprocessor(scale=False)
    out = pp.fit_transform(df, y)
    # Row 2 is FLAG_OWN_CAR=1 with missing age - should get median of [5, 10] = 7.5
    assert out.loc[2, "OWN_CAR_AGE"] == pytest.approx(7.5)
