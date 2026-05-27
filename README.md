# FinWatch: Consumer Financial Vulnerability Early Warning System

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)](https://lightgbm.readthedocs.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-teal)](https://fastapi.tiangolo.com/)
[![FCA Consumer Duty](https://img.shields.io/badge/FCA-Consumer%20Duty%202023-navy)](https://www.fca.org.uk/firms/consumer-duty)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage: 70%+](https://img.shields.io/badge/coverage-70%25+-brightgreen)](tests/)

Traditional credit models answer one question: will this new applicant default? I built FinWatch to answer a different one - which of our **existing** customers is silently struggling right now, and what should we do about it?

This matters because the **FCA Consumer Duty (July 2023)** creates a legal obligation for banks to proactively identify customers in vulnerable circumstances - not wait for them to miss a payment. FinWatch is a full production ML system that continuously scores existing customers for financial vulnerability and routes them to tiered interventions before they reach crisis point.

---

## Table of Contents

- [The Problem](#the-problem)
- [System Architecture](#system-architecture)
- [Data](#data)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Feature Engineering](#feature-engineering)
- [Modelling](#modelling)
- [Fairness and Compliance](#fairness-and-compliance)
- [Intervention Tiers](#intervention-tiers)
- [Serving Layer](#serving-layer)
- [Monitoring](#monitoring)
- [Dashboard](#dashboard)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Tech Stack](#tech-stack)

---

## The Problem

Credit risk modelling has a blind spot. It focuses entirely on the application stage - once a loan is approved, most banks know very little about how a customer's financial position evolves over time. By the time distress signals appear in payment data, it is often too late to intervene meaningfully.

A few things make this problem genuinely hard:

- **Vulnerability is gradual.** A customer who was comfortably managing a mortgage at 2% may be severely stretched at 5.5%. The shift happens across months, not overnight.
- **Macro shocks matter.** Energy price spikes, inflation, and base rate rises hit different customer segments differently. A model trained on pre-2022 data has never seen conditions like 2022-2024.
- **Class imbalance is severe.** Vulnerable customers are a minority. Standard accuracy metrics will flatter a model that simply labels everyone as fine.
- **Fairness is a legal requirement.** The FCA expects banks to demonstrate that automated systems do not treat protected groups disproportionately.

FinWatch addresses all four of these.

---

## System Architecture

```
DATA SOURCES
  Home Credit Bureau Data  +  ONS API (Live Macro Indicators)
              |
              v
      TRAINING PIPELINE
  CustomerPreprocessor  ->  Feature Engineering
          |
  SMOTE (train fold only)  ->  3 Model Candidates
          |                    LR baseline, LightGBM, XGBoost
  Optuna HPO (50 trials)
          |
  Champion Selection (PR-AUC)  ->  Fairness Gate (DIR >= 0.80)
          |
  MLflow Registry  ->  Threshold Calibration
              |
              v
        SERVING LAYER
  FastAPI /score/single  ->  InterventionEngine
          /score/batch       (ESCALATE / OUTREACH / MONITOR)
          /health            + SHAP explanations
              |
     ---------+---------
     |                 |
  MONITORING       DASHBOARD
  PSI Drift        Portfolio overview
  Macro Drift      Score lookup
  Fairness Log     Drift monitor
  -> Retrain flag  Fairness audit log
```

---

## Data

I used the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset as the core customer feature set. It contains 307,511 loan applications with 122 features covering income, credit amounts, employment history, external credit scores, and demographic attributes.

The target variable is binary: 1 = client with payment difficulties, 0 = performing client. The class imbalance is approximately 8:92, which shaped every modelling decision downstream.

On top of the Home Credit features, I append five live macroeconomic indicators pulled from the **ONS (Office for National Statistics) API** at scoring time:

| Indicator | Why it matters |
|-----------|----------------|
| Bank of England base rate | Directly affects variable mortgage and loan repayment costs |
| CPI (inflation) | Erodes real disposable income |
| Unemployment rate | Signals labour market stress |
| Household energy expenditure | A major fixed cost that squeezes budgets during energy shocks |
| GfK consumer confidence | Forward-looking sentiment indicator |

These are appended as constant columns to every row - every customer is scored against the same current macroeconomic environment. The baseline values at training time are saved, and drift detection flags if any indicator moves more than 1.5 standard deviations from that baseline.

---

## Exploratory Data Analysis

Before writing a single line of preprocessing or modelling code, I ran a full EDA in `notebooks/01_EDA.ipynb`. The findings that shaped the rest of the project:

**Target distribution.** 8.07% positive rate confirms severe class imbalance. ROC-AUC will be misleading here - I chose PR-AUC as the primary evaluation metric.

**Missing values.** `EXT_SOURCE_2` is 8% missing, `AMT_GOODS_PRICE` is 0.3% missing. Rates are low enough that median/mode imputation is appropriate.

**External credit scores.** `EXT_SOURCE_2` shows the clearest separation between target classes of any individual feature.

**Financial ratios.** Raw credit amounts are not particularly predictive. The ratio of credit to income, and of annuity to income, carry far more signal.

**Protected attributes.** The gender split is approximately 66% female, 34% male. Checking disparate impact at the EDA stage confirmed no obviously discriminatory signals before modelling.

---

## Preprocessing

The `CustomerPreprocessor` class in `finwatch/preprocessor.py` handles all data cleaning. I implemented it as a fitted transformer - it learns statistics from training data during `.fit()`, saves them, and applies those exact saved values at scoring time via `.transform()`.

This distinction matters. A script that recalculates statistics from inference-time data would behave differently at test time than at training time, making reported metrics unreliable and production behaviour unpredictable.

Steps in order:

1. **Missing value imputation** - numeric columns filled with training median, categoricals with training mode
2. **Categorical encoding** - binary categoricals label-encoded, multi-value categoricals one-hot encoded
3. **Numeric scaling** - StandardScaler fitted on training set only
4. **Macro feature append** - five ONS indicators appended after all other transformations

The fitted preprocessor is serialised to `data/processed/preprocessor.pkl` after training and loaded at API startup. The model and preprocessor always move together - they must be version-matched.

---

## Feature Engineering

Raw columns often carry less signal than their relationships with each other. `finwatch/features.py` creates three categories of derived features:

**Financial stress ratios:**
- `credit_to_income_ratio` - how many years of gross salary is the debt?
- `annuity_to_income_ratio` - what fraction of income goes to loan repayments?
- `goods_to_credit_ratio` - how much of the credit was for goods vs administrative costs?

**Employment stability:**
- `days_employed_pct_of_age` - what proportion of your life have you been employed? A 50-year-old with 2 years of employment tells a different story than a 25-year-old in the same position. Raw DAYS_EMPLOYED cannot capture this.

**Binary flags:**
- `is_high_credit_to_income` - threshold flag at credit/income > 5
- `is_young_borrower` - age under 30

I kept the feature set focused rather than exhaustive. On a regulated system, every feature needs to be justifiable to a compliance team.

---

## Modelling

### Why gradient boosting over deep learning

Deep learning was not in contention. The literature is clear that on tabular datasets without spatial or sequential structure, gradient boosting consistently outperforms neural networks (Grinsztajn et al., 2022). The Home Credit dataset sits comfortably within the regime where gradient boosting dominates. It also trains much faster and produces SHAP explanations natively - non-negotiable for a regulated deployment.

### Handling class imbalance

I used SMOTE (Synthetic Minority Over-sampling Technique) rather than class weighting alone. Class weighting adjusts the loss function but the model still sees only those same 8% of real examples. SMOTE generates synthetic minority examples by interpolating between existing ones in feature space, giving the model more diverse examples to learn from.

**Critical constraint:** SMOTE is applied only to the training fold, never before the train/validation split. Applying it earlier would let synthetic examples contaminate the validation set, making performance metrics unreliable.

### Hyperparameter optimisation

I used Optuna with TPE (Tree-structured Parzen Estimator) Bayesian optimisation for 50 trials on both LightGBM and XGBoost. TPE builds a probabilistic model of which hyperparameter regions produce good results and focuses search there - far more efficient than random or grid search.

### Champion selection

All three candidates are compared on PR-AUC on the validation set. I chose PR-AUC over ROC-AUC because ROC-AUC is inflated by the majority class. A model that labels everyone as fine will still achieve a high ROC-AUC. PR-AUC cannot be fooled this way - it focuses entirely on how well the model identifies the vulnerable minority.

### Probability calibration and threshold optimisation

Raw model outputs are calibrated with Platt scaling (CalibratedClassifierCV). I then sweep every threshold from 0.01 to 0.99, computing:

```
cost = (false_negatives * 5) + (false_positives * 1)
```

The 5:1 asymmetry reflects the regulatory reality: missing a vulnerable customer is substantially more harmful than flagging someone who turns out to be fine. The threshold minimising this cost is saved with the model artefacts.

---

## Fairness and Compliance

Before any model is registered to MLflow, it must pass the **FCA 4/5ths disparate impact test**. This is a hard gate - a failing model raises a `ValueError` and is never deployed.

The Disparate Impact Ratio (DIR) is computed for gender, family status, and education type:

```
DIR = (ESCALATE rate of disadvantaged group) / (ESCALATE rate of most-advantaged group)
```

Threshold: DIR >= 0.80. Monthly fairness audit results are logged to `data/processed/fairness_log.parquet`.

This system also satisfies **UK GDPR Article 22** via SHAP values - every individual scoring decision is accompanied by the top contributing features and their direction of influence.

---

## Intervention Tiers

| Score | Tier | Action |
|-------|------|--------|
| >= 0.70 | ESCALATE | Immediate outreach - assign to specialist vulnerability team |
| 0.40 - 0.70 | OUTREACH | Proactive contact - offer support products, payment holidays |
| < 0.40 | MONITOR | No action required - continue standard monitoring |

Thresholds are calibrated per model run, not hardcoded.

---

## Serving Layer

The model is served via FastAPI in `api/main.py`. Three endpoints:

- `POST /score/single` - score one customer, returns tier, score, and top SHAP features
- `POST /score/batch` - score a list of customers
- `GET /health` - API status and current model version

Request validation is handled by Pydantic v2 schemas. Invalid inputs are rejected before reaching the model. Swagger docs auto-generated at `/docs`.

---

## Monitoring

`monitoring/psi_monitor.py` implements PSI drift detection:

| PSI | Interpretation | Action |
|-----|----------------|--------|
| < 0.10 | No significant change | None |
| 0.10 - 0.20 | Moderate shift | Monitor |
| > 0.20 | Significant shift | Review / retrain |

Three retraining triggers:
1. **Scheduled** - monthly baseline retrain
2. **Feature drift** - PSI > 0.20 on any key feature
3. **Macro drift** - any ONS indicator moves more than 1.5 standard deviations from training baseline

---

## Dashboard

The Streamlit dashboard in `dashboard/app.py` has four pages:

- **Portfolio Overview** - KPI cards, tier donut chart, score distribution histogram
- **Score a Customer** - live API call with SHAP explanation chart
- **Drift Monitor** - PSI values per feature with threshold reference lines
- **Fairness Audit** - DIR trend over time, audit log

Includes demo mode with synthetic data when no model artefacts exist yet.

---

## Results

*To be updated after training. Will include PR-AUC on OOT test set, calibrated threshold, confusion matrix, fairness audit results, and tier distribution.*

---

## Project Structure

```
FinWatch/
|-- finwatch/                   # Core Python package
|   |-- constants.py
|   |-- preprocessor.py         # CustomerPreprocessor
|   |-- features.py             # Feature engineering
|   |-- macro_data.py           # ONS API client
|   |-- models.py               # Training + Optuna HPO
|   |-- decision_engine.py      # InterventionEngine
|   |-- fairness.py             # Disparate impact audit
|   |-- explainability.py       # SHAP wrapper
|
|-- training/train.py           # End-to-end CLI training pipeline
|-- api/main.py                 # FastAPI serving layer
|-- api/schemas.py              # Pydantic request/response models
|-- monitoring/psi_monitor.py   # PSI drift detection
|-- dashboard/app.py            # Streamlit dashboard
|-- tests/                      # pytest suite (70%+ coverage gate)
|-- notebooks/01_EDA.ipynb
|-- docker-compose.yml
|-- requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/ouyale/finwatch.git
cd finwatch
pip install -r requirements.txt
```

Download `application_train.csv` from [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) and place in `data/raw/`, then:

```bash
python training/train.py --data-path data/raw/application_train.csv --experiment finwatch-v1 --n-trials 50
docker-compose up --build
```

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Dashboard | http://localhost:8501 |

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Core ML | LightGBM + XGBoost | Production standard for tabular financial data |
| HPO | Optuna (TPE) | Bayesian search, 50 trials with pruning |
| Calibration | Platt Scaling | Honest probabilities for threshold optimisation |
| Imbalance | SMOTE (train fold only) | Diverse minority examples without leakage |
| Fairness | 4/5ths rule (DIR >= 0.80) | FCA-adopted disparate impact threshold |
| Explainability | SHAP TreeExplainer | Exact explanations satisfying UK GDPR Art. 22 |
| Drift | PSI | Industry-standard feature drift metric |
| Experiment tracking | MLflow | Model registry and artefact storage |
| API | FastAPI + Pydantic v2 | Auto-generated docs, schema validation |
| Dashboard | Streamlit + Plotly | Portfolio intelligence UI |
| Containerisation | Docker + docker-compose | Reproducible deployment |
| CI/CD | GitHub Actions | Lint + test + coverage gate on every push |
| Macro features | ONS API | Live UK economic indicators |

---

## Regulatory Context

| Regulation | Relevance |
|-----------|-----------|
| FCA Consumer Duty (2023) | Proactive identification of vulnerable customers - primary mandate |
| FCA PS21/11 | Fair treatment of vulnerable customers |
| UK GDPR Article 22 | Right to explanation for automated decisions |
| PRA SS1/23 | Model validation, monitoring, and challenger processes |

---

**Barbara Werobaobayi**
MSc Machine Learning and Deep Learning, University of Strathclyde
