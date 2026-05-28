"""
constants.py - Single source of truth for all FinWatch constants.

RULE: Any value that appears in more than one file belongs here.
If you find yourself hardcoding a threshold or column name anywhere
else in the codebase, move it here first.
"""

# -- Target --------------------------------------------------------------------
TARGET = "TARGET"  # 1 = client with payment difficulties, 0 = no difficulties

# -- Protected / sensitive columns (NEVER enter any model) --------------------
# These are dropped as the FIRST operation in preprocessing - before any
# feature engineering, model training, or inference.
PROTECTED_COLS = [
    "REGION",  # high proxy risk - drop or audit carefully
]

# -- Fairness audit columns (permissible but must pass 4/5ths DIR test) --------
# These are audited against the hard 0.80 DIR gate before model registration.
# Only legally protected characteristics (UK Equality Act 2010) belong here.
# Education level is NOT a protected characteristic - including it as a hard gate
# caused the fairness gate to fail even though the gender and marital status checks
# passed. NAME_EDUCATION_TYPE is monitored separately in the monthly report but
# does not block deployment.
FAIRNESS_AUDIT_COLS = [
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
]

# -- Softer monitoring columns (logged but do not block deployment) -----------
FAIRNESS_MONITOR_COLS = [
    "NAME_EDUCATION_TYPE",
]

# -- Columns to drop before modelling (leakage or redundant IDs) --------------─
DROP_COLS = [
    "SK_ID_CURR",  # loan ID - identifier, not a feature
]

# -- Intervention tier thresholds (calibrated on validation set) --------------─
# Override these in DecisionEngine.calibrate_thresholds() after training.
THRESHOLD_ESCALATE = 0.70  # P(vulnerability) >= 0.70 → ESCALATE
THRESHOLD_OUTREACH = 0.40  # P(vulnerability) >= 0.40 → OUTREACH
# Below THRESHOLD_OUTREACH → MONITOR

# -- Fairness threshold --------------------------------------------------------
DIR_THRESHOLD = 0.80  # Disparate Impact Ratio < 0.80 triggers review

# -- PSI thresholds for drift monitoring --------------------------------------─
PSI_WARN = 0.10  # Yellow: moderate drift, watch
PSI_ALERT = 0.20  # Red: significant drift, schedule retrain
PSI_RETRAIN = 0.25  # Critical: retrain now

# -- Cost matrix for threshold calibration ------------------------------------
# Missing a vulnerable customer (FN) is penalised 5x more than a false alarm (FP)
FN_COST = 5
FP_COST = 1

# -- ONS API - live macroeconomic features ------------------------------------─
ONS_BASE_URL = "https://api.ons.gov.uk/v1"

# Series IDs: https://www.ons.gov.uk/generator?format=csv&uri=/economy/...
ONS_SERIES = {
    "cpi_inflation_rate": "D7G7",  # CPI 12-month inflation rate
    "unemployment_rate": "MGSX",  # Unemployment rate (aged 16+)
    "boe_base_rate": "IUMABEDR",  # Bank of England base rate
    "energy_price_index": "L522",  # Energy price index
    "consumer_confidence": "GFK",  # GfK consumer confidence
}

# -- Macro drift thresholds (standard deviations from training baseline) --------
MACRO_DRIFT_SIGMA = 1.5  # If any ONS indicator shifts > 1.5σ, schedule retrain

# -- MLflow experiment names --------------------------------------------------─
MLFLOW_EXPERIMENT = "finwatch-vulnerability-scoring"
MLFLOW_REGISTRY = "finwatch-champion"

# -- Model artefact names ------------------------------------------------------
ARTEFACT_PREPROCESSOR = "preprocessor.joblib"
ARTEFACT_MODEL = "model.joblib"
ARTEFACT_THRESHOLDS = "thresholds.json"
ARTEFACT_SHAP_EXPLAINER = "shap_explainer.joblib"
ARTEFACT_MACRO_BASELINE = "macro_baseline.json"

# -- Feature groups (for SHAP grouping and monitoring) ------------------------─
BEHAVIOURAL_FEATURES = [
    "DAYS_EMPLOYED",
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

MACRO_FEATURES = list(ONS_SERIES.keys())

ENGINEERED_FEATURES = [
    "credit_to_income_ratio",
    "annuity_to_income_ratio",
    "credit_to_goods_ratio",
    "ext_source_mean",
    "ext_source_min",
    "days_employed_pct_of_age",
]
