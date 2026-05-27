"""
schemas.py - Pydantic request/response models for the FinWatch API.

Every field has a description so the auto-generated FastAPI docs
(/docs) are readable without needing to look at the source code.
"""

from typing import List, Optional
from pydantic import BaseModel, Field

# -- Request ------------------------------------------------------------------─


class CustomerRecord(BaseModel):
    """A single customer's application and behavioural data."""

    SK_ID_CURR: int = Field(..., description="Unique customer identifier")
    AMT_INCOME_TOTAL: float = Field(..., description="Annual income (£)")
    AMT_CREDIT: float = Field(..., description="Total credit outstanding (£)")
    AMT_ANNUITY: float = Field(..., description="Monthly loan annuity (£)")
    AMT_GOODS_PRICE: float = Field(..., description="Price of goods for which loan was given")
    CODE_GENDER: str = Field(..., description="M or F")
    DAYS_BIRTH: int = Field(..., description="Days since birth (negative integer)")
    DAYS_EMPLOYED: int = Field(..., description="Days since last employment start")
    NAME_INCOME_TYPE: str = Field(..., description="Working / Pensioner / State servant / etc.")
    NAME_EDUCATION_TYPE: str = Field(..., description="Highest education level")
    NAME_FAMILY_STATUS: str = Field(..., description="Married / Single / etc.")
    EXT_SOURCE_1: Optional[float] = Field(
        None, description="External credit score 1 (may be missing)"
    )
    EXT_SOURCE_2: Optional[float] = Field(
        None, description="External credit score 2 (may be missing)"
    )
    EXT_SOURCE_3: Optional[float] = Field(
        None, description="External credit score 3 (may be missing)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "SK_ID_CURR": 100001,
                "AMT_INCOME_TOTAL": 202500.0,
                "AMT_CREDIT": 406597.5,
                "AMT_ANNUITY": 24700.5,
                "AMT_GOODS_PRICE": 351000.0,
                "CODE_GENDER": "M",
                "DAYS_BIRTH": -9461,
                "DAYS_EMPLOYED": -637,
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Single / not married",
                "EXT_SOURCE_1": 0.502,
                "EXT_SOURCE_2": 0.262,
                "EXT_SOURCE_3": None,
            }
        }


class BatchRequest(BaseModel):
    """Batch of customer records for portfolio scoring."""

    customers: List[CustomerRecord] = Field(..., description="List of customer records to score")


# -- Response ------------------------------------------------------------------


class SHAPFeature(BaseModel):
    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP value - positive increases risk")
    direction: str = Field(..., description="increases_risk or reduces_risk")


class ScoreResponse(BaseModel):
    """Vulnerability score and intervention decision for one customer."""

    customer_id: int = Field(..., description="Customer identifier")
    vulnerability_score: float = Field(..., description="Calibrated P(vulnerable), 0–1")
    tier: str = Field(..., description="MONITOR / OUTREACH / ESCALATE")
    threshold_escalate: float = Field(..., description="Threshold used for ESCALATE")
    threshold_outreach: float = Field(..., description="Threshold used for OUTREACH")
    top_shap_features: List[SHAPFeature] = Field(..., description="Top features driving this score")
    model_version: str = Field(..., description="Model version used for scoring")
    scored_at: str = Field(..., description="UTC timestamp of scoring")
    macro_snapshot: dict = Field(..., description="ONS macro indicators used at scoring time")


class BatchResponse(BaseModel):
    """Batch scoring results with portfolio summary."""

    results: List[ScoreResponse]
    total_customers: int
    escalate_count: int
    outreach_count: int
    monitor_count: int
    scored_at: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    macro_fresh: bool
