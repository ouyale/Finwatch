"""
preprocessor.py - CustomerPreprocessor

Sklearn-compatible transformer that cleans and encodes Home Credit
application data for consumer retail banking vulnerability scoring.

Usage
-----
    preprocessor = CustomerPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train, y_train)
    X_val_clean   = preprocessor.transform(X_val)

    # Single-record inference (FastAPI endpoint)
    record = preprocessor.transform_single({"AMT_INCOME_TOTAL": 50000, ...})
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from .constants import DROP_COLS, PROTECTED_COLS, TARGET

logger = logging.getLogger(__name__)


class CustomerPreprocessor(BaseEstimator, TransformerMixin):
    """
    End-to-end preprocessing for Home Credit consumer vulnerability data.

    Steps (in order - do NOT reorder):
    1. Drop protected / leakage columns  ← ALWAYS FIRST
    2. Deduplicate on SK_ID_CURR
    3. Fix impossible values
    4. Encode categoricals
    5. Handle missing values (sentinel + flag for credit bureau scores)
    6. Scale numerics
    7. Align columns to training schema
    """

    def __init__(
        self,
        drop_cols: list = None,
        protected_cols: list = None,
        scale: bool = True,
    ):
        self.drop_cols = drop_cols or DROP_COLS
        self.protected_cols = protected_cols or PROTECTED_COLS
        self.scale = scale

        # Fitted state - populated during .fit()
        self._feature_names: list = []
        self._cat_encodings: dict = {}
        self._numeric_medians: dict = {}
        self._scaler: Optional[StandardScaler] = None
        self._fitted: bool = False

    # -- Fit ------------------------------------------------------------------─

    def fit(self, X: pd.DataFrame, y=None) -> "CustomerPreprocessor":
        """Learn statistics from TRAINING DATA ONLY."""
        logger.info("CustomerPreprocessor.fit() - %d rows, %d cols", *X.shape)

        df = X.copy()
        df = self._drop_poison(df)
        df = self._fix_impossible_values(df)
        df = self._encode_categoricals(df, fit=True)
        df = self._handle_missing(df, fit=True)

        if self.scale:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._scaler = StandardScaler()
            self._scaler.fit(df[numeric_cols])
            self._scaler_cols = numeric_cols

        self._feature_names = df.columns.tolist()
        self._fitted = True
        logger.info("Fit complete. %d features retained.", len(self._feature_names))
        return self

    # -- Transform ------------------------------------------------------------─

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to a dataframe."""
        self._check_fitted()
        df = X.copy()
        df = self._drop_poison(df)
        df = self._fix_impossible_values(df)
        df = self._encode_categoricals(df, fit=False)
        df = self._handle_missing(df, fit=False)

        if self.scale and self._scaler is not None:
            cols = [c for c in self._scaler_cols if c in df.columns]
            df[cols] = self._scaler.transform(df[cols])

        # Align to training schema: add missing cols as 0, drop extra cols
        for col in self._feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self._feature_names]

        return df

    def transform_single(self, record: dict) -> pd.DataFrame:
        """
        Transform a single application record for real-time inference.

        Parameters
        ----------
        record : dict  Raw application fields as key-value pairs.

        Returns
        -------
        pd.DataFrame  Single-row DataFrame ready for model.predict_proba()
        """
        return self.transform(pd.DataFrame([record]))

    # -- Private helpers ------------------------------------------------------─

    def _drop_poison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 1 - ALWAYS FIRST. Drop protected and leakage columns."""
        poison = [c for c in self.drop_cols + self.protected_cols if c in df.columns]
        if poison:
            logger.debug("Dropping %d poison columns: %s", len(poison), poison)
        return df.drop(columns=poison, errors="ignore")

    def _fix_impossible_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Step 3 - Replace impossible values with NaN."""
        # DAYS_BIRTH: stored as negative days, so positive values are wrong
        if "DAYS_BIRTH" in df.columns:
            age_years = df["DAYS_BIRTH"].abs() / 365
            df.loc[(age_years < 18) | (age_years > 100), "DAYS_BIRTH"] = np.nan

        # DAYS_EMPLOYED: 365243 is a sentinel for 'unemployed/N/A' in this dataset
        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED_IS_NA"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

        # AMT_INCOME_TOTAL: negative income is impossible
        if "AMT_INCOME_TOTAL" in df.columns:
            df.loc[df["AMT_INCOME_TOTAL"] <= 0, "AMT_INCOME_TOTAL"] = np.nan

        return df

    def _encode_categoricals(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Step 4 - Encode categorical columns."""
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col in cat_cols:
            if fit:
                mapping = {
                    v: i for i, v in enumerate(sorted(df[col].dropna().unique()))
                }
                mapping["__unknown__"] = -1
                self._cat_encodings[col] = mapping
            enc = self._cat_encodings.get(col, {})
            df[col] = df[col].map(enc).fillna(-1).astype(int)

        return df

    def _handle_missing(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        Step 5 - Handle missing values.

        External credit bureau scores (EXT_SOURCE_1/2/3) have high missingness
        (~50-65%). A missing bureau score is itself a predictive signal. Strategy:
          - Add binary flag: has_ext_source_{n}
          - Fill missing with sentinel value -1 (not median)
        """
        ext_sources = [
            c
            for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
            if c in df.columns
        ]
        for col in ext_sources:
            df[f"has_{col.lower()}"] = df[col].notna().astype(int)
            df[col] = df[col].fillna(-1)

        # All other numeric columns: fill with training median
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].isna().any():
                if fit:
                    self._numeric_medians[col] = df[col].median()
                median = self._numeric_medians.get(col, 0)
                df[col] = df[col].fillna(median)

        return df

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "CustomerPreprocessor has not been fitted. Call .fit() first."
            )

    # -- Properties ------------------------------------------------------------

    @property
    def feature_names(self) -> list:
        """Return list of feature names after fitting."""
        self._check_fitted()
        return self._feature_names
