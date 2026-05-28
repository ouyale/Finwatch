"""
main.py - FinWatch FastAPI inference service.

Endpoints
---------
  GET  /health           - liveness + model version check
  POST /score/single     - score one customer, returns full explanation
  POST /score/batch      - score a portfolio, returns tier summary

Run locally
-----------
    uvicorn api.main:app --reload --port 8000

Then open: http://localhost:8000/docs  (interactive Swagger UI)
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from api.schemas import (
    BatchRequest,
    BatchResponse,
    CustomerRecord,
    HealthResponse,
    ScoreResponse,
    SHAPFeature,
)
from finwatch.constants import (
    ARTEFACT_MODEL,
    ARTEFACT_PREPROCESSOR,
    ARTEFACT_SHAP_EXPLAINER,
)
from finwatch.decision_engine import InterventionEngine
from finwatch.explainability import explain_single
from finwatch.features import run_all
from finwatch.macro_data import get_macro_snapshot


def _align_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """
    Align a feature DataFrame to exactly the columns the model was trained on.

    The training pipeline runs preprocessor -> feature engineering -> fillna(0)
    before the model sees the data. At inference time the same steps run, but
    the incoming customer record may have fewer raw columns than the full
    training dataset. This function handles any remaining gap after feature
    engineering - it adds missing columns as 0, drops unexpected columns,
    and reorders to match the training column order exactly.

    feature_names is loaded from data/processed/feature_names.json at startup.
    """
    X = X.fillna(0)
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names]

logger = logging.getLogger(__name__)

# -- Global model state (loaded once on startup) ------------------------------─
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts on startup. Release on shutdown."""
    artefact_dir = Path("data/processed")
    logger.info("Loading model artefacts from %s...", artefact_dir)

    _state["preprocessor"] = joblib.load(artefact_dir / ARTEFACT_PREPROCESSOR)
    _state["model"] = joblib.load(artefact_dir / ARTEFACT_MODEL)
    _state["explainer"] = joblib.load(artefact_dir / ARTEFACT_SHAP_EXPLAINER)
    _state["engine"] = InterventionEngine.load_thresholds(str(artefact_dir / "thresholds.json"))
    _state["model_version"] = "0.1.0"  # TODO: read from MLflow registry

    # Load the exact feature columns the model was trained on.
    # Saved by training/train.py as data/processed/feature_names.json.
    feature_names_path = artefact_dir / "feature_names.json"
    with open(feature_names_path) as f:
        _state["feature_names"] = json.load(f)
    logger.info("Loaded %d feature names.", len(_state["feature_names"]))

    logger.info(
        "Model loaded. Thresholds: ESCALATE=%.2f, OUTREACH=%.2f",
        _state["engine"].threshold_escalate,
        _state["engine"].threshold_outreach,
    )
    yield
    _state.clear()


app = FastAPI(
    title="FinWatch - Consumer Vulnerability Early Warning API",
    description=(
        "Scores bank customers for financial vulnerability using behavioural "
        "features and live ONS macroeconomic indicators. Returns calibrated "
        "probability scores, tiered intervention decisions (MONITOR / OUTREACH / ESCALATE), "
        "and SHAP feature explanations. FCA Consumer Duty aligned."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# -- Endpoints ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health():
    """Liveness check - confirms model is loaded and macro data is fresh."""
    return HealthResponse(
        status="ok",
        model_version=_state.get("model_version", "unknown"),
        macro_fresh=True,  # TODO: check cache timestamp
    )


@app.post("/score/single", response_model=ScoreResponse, tags=["Scoring"])
async def score_single(record: CustomerRecord):
    """
    Score a single customer for financial vulnerability.

    Returns a calibrated probability score, intervention tier,
    and the top 5 SHAP features explaining the score.
    """
    try:
        df = pd.DataFrame([record.model_dump()])
        macro_snapshot = get_macro_snapshot()

        # Preprocess → feature engineer → align → score
        X_clean = _state["preprocessor"].transform(df)
        X_feat = run_all(X_clean, macro_snapshot)
        X_feat = _align_features(X_feat, _state["feature_names"])

        probability = float(_state["model"].predict_proba(X_feat)[:, 1][0])

        # SHAP explanation
        shap_features = explain_single(
            _state["explainer"],
            X_feat,
            _state["feature_names"],
            n_features=5,
        )

        decision = _state["engine"].predict_single(
            customer_id=str(record.SK_ID_CURR),
            probability=probability,
            shap_features=shap_features,
            model_version=_state["model_version"],
        )

        return ScoreResponse(
            customer_id=record.SK_ID_CURR,
            vulnerability_score=decision.vulnerability_score,
            tier=decision.tier,
            threshold_escalate=decision.threshold_escalate,
            threshold_outreach=decision.threshold_outreach,
            top_shap_features=[SHAPFeature(**f) for f in shap_features],
            model_version=decision.model_version,
            scored_at=decision.scored_at,
            macro_snapshot=macro_snapshot,
        )

    except Exception as e:
        logger.exception("Error scoring customer %s", record.SK_ID_CURR)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchResponse, tags=["Scoring"])
async def score_batch(request: BatchRequest):
    """
    Score a portfolio of customers in a single request.

    Returns per-customer scores and a portfolio-level tier summary.
    Macro snapshot is fetched once and applied to all records.
    """
    try:
        macro_snapshot = get_macro_snapshot()
        results = []

        for record in request.customers:
            df = pd.DataFrame([record.model_dump()])
            X_clean = _state["preprocessor"].transform(df)
            X_feat = run_all(X_clean, macro_snapshot)
            X_feat = _align_features(X_feat, _state["feature_names"])
            probability = float(_state["model"].predict_proba(X_feat)[:, 1][0])

            shap_features = explain_single(
                _state["explainer"],
                X_feat,
                _state["feature_names"],
            )
            decision = _state["engine"].predict_single(
                customer_id=str(record.SK_ID_CURR),
                probability=probability,
                shap_features=shap_features,
                model_version=_state["model_version"],
            )
            results.append(
                ScoreResponse(
                    customer_id=record.SK_ID_CURR,
                    vulnerability_score=decision.vulnerability_score,
                    tier=decision.tier,
                    threshold_escalate=decision.threshold_escalate,
                    threshold_outreach=decision.threshold_outreach,
                    top_shap_features=[SHAPFeature(**f) for f in shap_features],
                    model_version=decision.model_version,
                    scored_at=decision.scored_at,
                    macro_snapshot=macro_snapshot,
                )
            )

        tiers = [r.tier for r in results]
        return BatchResponse(
            results=results,
            total_customers=len(results),
            escalate_count=tiers.count("ESCALATE"),
            outreach_count=tiers.count("OUTREACH"),
            monitor_count=tiers.count("MONITOR"),
            scored_at=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        logger.exception("Error in batch scoring")
        raise HTTPException(status_code=500, detail=str(e))
