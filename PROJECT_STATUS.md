# FinWatch - Project Status Document

**Author:** Barbara Werobaobayi  
**Programme:** MSc Machine Learning and Deep Learning, University of Strathclyde  
**Last updated:** May 2026 (session 2)  
**Purpose:** Running record of what is built, what is working, what is next, and what decisions were made and why. Update this every session.

---

## Where We Are Right Now

The project has a complete codebase and a green CI pipeline. The two preprocessor gaps identified in the post-EDA review have been fixed. The training pipeline has NOT been run yet. No real model metrics exist. The next major milestone is running training and getting real numbers.

**Immediate next step:** `python training/train.py --data-path data/raw/application_train.csv --experiment finwatch-v1 --n-trials 50`

---

## What Is Built

### Core Package - `finwatch/`

| File | What it does | Status |
|---|---|---|
| `constants.py` | Single source of truth for all thresholds, column names, API URLs | Complete |
| `preprocessor.py` | CustomerPreprocessor - cleans, encodes, scales training data | Complete, reviewed post-EDA |
| `features.py` | Feature engineering - financial ratios, document flags, enquiry counts, macro features | Complete, bug fixed |
| `models.py` | Trains LogisticRegression, LightGBM, XGBoost. Selects champion by PR-AUC | Complete |
| `fairness.py` | Disparate Impact Ratio audit. Hard gate - model fails = not deployed | Complete |
| `decision_engine.py` | Converts probability scores to ESCALATE/OUTREACH/MONITOR tiers | Complete |
| `explainability.py` | SHAP TreeExplainer for feature attribution (UK GDPR Article 22 compliance) | Complete |
| `macro_data.py` | Fetches live ONS economic indicators (CPI, unemployment, BoE rate etc) | Complete |

### Other Layers

| Location | What it does | Status |
|---|---|---|
| `api/main.py` + `api/schemas.py` | FastAPI REST endpoint. POST /score returns tier + SHAP explanation | Complete, not yet tested |
| `training/train.py` | End-to-end training pipeline. MLflow tracking. Fairness gate. Saves artefacts | Complete, not yet run |
| `monitoring/psi_monitor.py` | Population Stability Index - detects data drift post-deployment | Complete |
| `dashboard/` | Streamlit dashboard for vulnerability monitoring | Complete, not yet tested |
| `docker-compose.yml` | Runs API + MLflow + dashboard together in containers | Complete, not yet tested |
| `notebooks/01_EDA.ipynb` | Full exploratory data analysis with real numbers from Kaggle data | Complete and run |

### Tests

| File | What it covers | Tests |
|---|---|---|
| `tests/test_preprocessor.py` | CustomerPreprocessor - all steps, deduplication, OWN_CAR_AGE, edge cases | 11 tests |
| `tests/test_fairness.py` | DIR audit, fairness gate pass/fail, edge cases | 5 tests |
| `tests/test_decision_engine.py` | Tier assignment, threshold calibration, batch scoring | 6 tests |
| `tests/test_features.py` | All feature engineering functions, edge cases, zero division | 22 tests |

**Total: 44 tests, all passing. CI green (pending push with session 2 changes).**

### Infrastructure

| Item | Status |
|---|---|
| GitHub repository (ouyale/Finwatch) | Live |
| CI pipeline (GitHub Actions) | Green - 6 runs to get here |
| pyproject.toml packaging | Complete |
| `.flake8` config | Complete |
| `.gitignore` | Complete - data/raw excluded |

---

## CI History (What We Fixed and Why)

Run 1 - Black formatting failure. Fixed by running `black` locally on all files.  
Run 2 - Flake8 failures. Unused imports across multiple files, f-string with no variables, imports after sys.path.insert. Fixed each one.  
Run 3 - Package not importable in CI. `finwatch` wasn't installed in the CI environment. Fixed by creating `pyproject.toml` and adding `pip install -e .` to ci.yml.  
Run 4 - Still red because pyproject.toml used `setuptools.backends.legacy:build` which requires setuptools 68+. Fixed by switching to `setuptools.build_meta`.  
Run 5 - Test failure: `test_days_employed_sentinel_handled`. StandardScaler was scaling the binary `DAYS_EMPLOYED_IS_NA` column (0/1 → 3.0). Fixed by excluding flag columns from scaler.  
Run 6 - Coverage gate: 45.51% but threshold is 70%. Added `tests/test_features.py` (22 tests). Omitted `macro_data.py` and `models.py` from coverage (live HTTP calls and model training belong in integration tests, not unit tests). Also fixed a real operator precedence bug in `features.py` that the tests caught.  
Run 6 - GREEN. 40 tests passing, coverage 82%.

---

## EDA Findings (Real Numbers from Kaggle Data)

Ran on `application_train.csv` - 307,511 rows, 122 columns.

**Class imbalance:** 11.4:1. Only 8.1% of customers are vulnerable. A model predicting "not vulnerable" for everyone would be 91.9% accurate and completely useless. This justifies SMOTE and PR-AUC as the evaluation metric.

**Missing data:**
- 67 columns have missing values
- EXT_SOURCE_1: 56.4% missing - MNAR (missing = thin credit file = vulnerability signal)
- EXT_SOURCE_2: 0.2% missing - MNAR
- EXT_SOURCE_3: 19.8% missing - MNAR
- Property columns (COMMONAREA, FLOORSMIN etc): ~70% missing - structural (renters have no property data)
- OWN_CAR_AGE: 66% missing - structural (FLAG_OWN_CAR=0 means no car exists)
- DAYS_EMPLOYED=365243: 55,374 rows (18%) - sentinel for unemployed

**MNAR confirmed for EXT_SOURCE:** Vulnerability rate when EXT_SOURCE_1 is missing is significantly higher than when it is present. The absence is a real signal - not random noise.

**Property columns missingness and NAME_HOUSING_TYPE:** Confirmed that COMMONAREA is ~100% missing for renters and people living with parents. NAME_HOUSING_TYPE already captures this with 6 categories each having different vulnerability rates. Creating a binary IS_RENTER column would collapse this nuance. No IS_RENTER column needed.

**Numeric skewness:** AMT_INCOME_TOTAL skewness = 392. Median imputation is required, not mean. A small number of very high earners pull the mean far above what is representative for most customers.

**Binary columns:** 33 binary (0/1) columns exist in the raw data. These must NOT be scaled - StandardScaler would convert 0→-0.3 and 1→3.0 which destroys the meaning of a binary indicator.

**Strongest predictors (all correlations < 0.20 - weak signals individually):**
1. EXT_SOURCE_3: -0.179
2. EXT_SOURCE_2: -0.161
3. EXT_SOURCE_1: -0.155
4. Age (DAYS_BIRTH): -0.078
5. REGION_RATING_CLIENT_W_CITY: +0.061

All correlations are weak. No single feature separates classes well. This is exactly why we need gradient boosting - it combines hundreds of weak signals into a strong prediction.

**Age finding:** Vulnerable customers are on average 3.4 years younger (40.8 vs 44.2). The vulnerable peak is around age 27-30. Financial stability increases with age.

**Fairness on raw data:**
- Male vulnerability rate: 10.1%
- Female vulnerability rate: 7.0%
- Gender DIR: 0.690 (fails the 0.80 FCA threshold)
- This reflects the underlying population, not model discrimination. The fairness gate in training ensures the MODEL output meets 0.80.

---

## Preprocessor Review (Post-EDA)

### What is handled correctly

- Protected columns dropped first (SK_ID_CURR, REGION)
- DAYS_BIRTH impossible values → NaN (age < 18 or > 100)
- DAYS_EMPLOYED=365243 sentinel → flag column + NaN
- Negative AMT_INCOME_TOTAL → NaN
- EXT_SOURCE missing → has_ext_source_N flag + sentinel -1 (not median)
- All other numeric missing → training median (justified by skewness analysis)
- Categorical encoding with unknown category handling (-1)
- StandardScaler excludes _IS_NA, has_, FLAG_*, and any 0/1 column
- Schema alignment at inference time (add missing cols as 0, reorder to training schema)

### Two gaps - both fixed in session 2

**Gap 1: Deduplication - FIXED.**  
Added `_deduplicate()` method. It runs as the very first step (before `_drop_poison`) because SK_ID_CURR is dropped in that step. The method calls `drop_duplicates(subset=["SK_ID_CURR"], keep="last")` and logs a warning if any duplicates were found. In the Kaggle training data there are no duplicates (confirmed by EDA), so this is a no-op during development. In production, where data arrives from multiple source systems, duplicate records would silently corrupt the fitted medians and encodings - this prevents that. The docstring step order was also corrected.

**Gap 2: OWN_CAR_AGE imputation - FIXED.**  
Added conditional imputation logic at the top of `_handle_missing`, before the generic median loop. Rows where FLAG_OWN_CAR=0 now receive OWN_CAR_AGE=0 (they have no car, so car age is zero). Rows where FLAG_OWN_CAR=1 and OWN_CAR_AGE is missing now receive the median age among car owners only (not the overall median). Because this runs before the generic loop, the generic loop finds no remaining NaN in OWN_CAR_AGE and leaves it alone.

---

## What Happens Next (In Order)

### 1. Commit and push session 2 changes
The two preprocessor fixes, four new tests, and updated status document all need to be committed and pushed to trigger CI. From the FinWatch directory:
```bash
git add -A
git commit -m "Fix preprocessor gaps: deduplication and OWN_CAR_AGE imputation"
git push
```

### 2. Run the training pipeline
```bash
cd /path/to/FinWatch
python training/train.py \
    --data-path data/raw/application_train.csv \
    --experiment finwatch-v1 \
    --n-trials 50
```
This will take 20-40 minutes. It trains three models (LogisticRegression, LightGBM, XGBoost), selects a champion by PR-AUC, runs the fairness gate, saves all artefacts, and logs everything to MLflow.

Expected outputs in `data/processed/`:
- `preprocessor.joblib`
- `model.joblib`
- `thresholds.json`
- `shap_explainer.joblib`
- `macro_baseline.json`

### 3. Update README with real metrics
The Results section in README.md currently has placeholder values. Fill in actual Test PR-AUC, Test ROC-AUC, champion model name, fairness DIR after training.

### 4. Test the API
```bash
docker-compose up --build
# Then open http://localhost:8000/docs
```
Submit a test customer record via the Swagger UI. Verify it returns a tier (ESCALATE/OUTREACH/MONITOR), a vulnerability score, and SHAP feature attributions.

### 5. Verify the full stack
- MLflow UI at http://localhost:5000 - check run is logged
- Streamlit dashboard at http://localhost:8501 - check it loads

### 6. Push final commit with real metrics
Commit the trained artefacts (NOT the raw data - that stays gitignored) and the updated README.

---

## Key Decisions and Their Justifications

| Decision | Justification |
|---|---|
| PR-AUC not accuracy as champion metric | 11.4:1 class imbalance makes accuracy meaningless. PR-AUC measures performance on the minority (vulnerable) class directly. |
| SMOTE on training fold only | Applying SMOTE before splitting would cause data leakage - synthetic points created from validation data would appear in training. |
| Gradient boosting over deep learning | Tabular data. All correlations < 0.20. Gradient boosting combines weak signals better than neural networks on structured tabular data at this size. |
| Median not mean for imputation | AMT_INCOME_TOTAL skewness = 392. Mean is pulled far above median by outliers and would impute unrealistically high values. |
| EXT_SOURCE: flag + sentinel not median | Vulnerability rate is significantly higher when EXT_SOURCE is missing. Missingness is itself a signal (thin credit file). Imputing with median would erase that signal. |
| No IS_RENTER binary column | NAME_HOUSING_TYPE already captures housing with 6 categories, each with different vulnerability rates. A binary column would lose nuance. |
| Binary columns excluded from StandardScaler | 33 binary columns in raw data. Scaling 0/1 to -0.3/3.0 destroys meaning. EDA confirmed this. Test caught the original bug. |
| FCA 4/5ths rule as hard gate | Regulatory requirement under Consumer Duty. A model that fails disparate impact is not deployed regardless of accuracy metrics. |
| SHAP for explanations | UK GDPR Article 22 requires explainability for automated decisions affecting individuals. SHAP provides legally defensible feature attributions. |
| macro_data.py and models.py omitted from unit test coverage | These require live HTTP calls and full model training respectively. They belong in integration tests. Omitting them is standard practice. |

---

## Files That Should NOT Be in GitHub

All of these are in `.gitignore`:

- `data/raw/` - 159MB Kaggle CSV. Never commit raw data.
- `data/processed/` - trained model artefacts. Too large, regenerated by training.
- `__pycache__/` and `*.pyc` - Python compiled files, machine-specific.
- `.env` - would contain API keys or secrets if we had any.
- `mlruns/` - MLflow experiment tracking data, generated locally.

---

## Concepts Still to Cover

These were mentioned as needed but not yet covered in depth:

- Python classes and functions line by line (beyond the preprocessor)
- Docker and docker-compose - what they actually do and why
- FastAPI and the API layer in detail
- MLflow - what experiment tracking means in practice
- Probability calibration (Platt scaling / CalibratedClassifierCV)
- SHAP mechanics in more depth
- PSI monitoring in practice
- How a real deployment pipeline differs from what we have here
