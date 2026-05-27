"""
train.py - End-to-end FinWatch training pipeline.

Run this to train, evaluate, and register a new model version.
Every run is tracked in MLflow. The model is only registered if it
passes the fairness gate (4/5ths DIR test).

Usage
-----
    python training/train.py
    python training/train.py --n-trials 100 --experiment finwatch-v2
    python training/train.py --skip-optuna   # fast run for testing
"""

import argparse
import logging

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from finwatch.constants import (
    ARTEFACT_MODEL,
    ARTEFACT_PREPROCESSOR,
    ARTEFACT_SHAP_EXPLAINER,
    ARTEFACT_THRESHOLDS,
    FAIRNESS_AUDIT_COLS,
    MLFLOW_EXPERIMENT,
    TARGET,
)
from finwatch.decision_engine import InterventionEngine
from finwatch.explainability import get_explainer
from finwatch.fairness import disparate_impact_audit, fairness_gate
from finwatch.features import run_all
from finwatch.macro_data import get_macro_snapshot, save_macro_baseline
from finwatch.models import (
    evaluate_model,
    select_champion,
    train_lightgbm,
    train_logistic_regression,
    train_xgboost,
)
from finwatch.preprocessor import CustomerPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("finwatch.train")


def load_data(data_path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows, %d columns", *df.shape)
    return df


def main(args):
    # -- 1. Load data ----------------------------------------------------------
    df = load_data(args.data_path)

    # Keep fairness columns aside before preprocessing (needed for audit)
    fairness_df = df[[c for c in FAIRNESS_AUDIT_COLS if c in df.columns]].copy()

    X = df.drop(columns=[TARGET])
    y = df[TARGET].values

    # -- 2. Split - 60% train / 20% val / 20% test ----------------------------
    # Stratified to preserve class imbalance in all splits
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
    )
    logger.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # -- 3. Preprocess --------------------------------------------------------─
    preprocessor = CustomerPreprocessor()
    X_train_clean = preprocessor.fit_transform(X_train, y_train)
    X_val_clean = preprocessor.transform(X_val)
    X_test_clean = preprocessor.transform(X_test)

    # -- 4. Feature engineering ------------------------------------------------
    logger.info("Fetching ONS macro snapshot...")
    macro_snapshot = get_macro_snapshot()

    X_train_feat = run_all(X_train_clean, macro_snapshot)
    X_val_feat = run_all(X_val_clean, macro_snapshot)
    X_test_feat = run_all(X_test_clean, macro_snapshot)

    # -- 5. SMOTE - training fold ONLY ----------------------------------------
    logger.info("Applying SMOTE to training fold only...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_feat, y_train)
    logger.info(
        "After SMOTE: %d rows (was %d). Class balance: %.1f%%",
        len(X_train_bal),
        len(X_train_feat),
        100 * y_train_bal.mean(),
    )

    # -- 6. Train models ------------------------------------------------------─
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name="finwatch-training"):
        mlflow.log_params(
            {
                "n_train": len(X_train_bal),
                "n_val": len(X_val_feat),
                "n_test": len(X_test_feat),
                "n_features": X_train_feat.shape[1],
                "optuna_trials": args.n_trials,
            }
        )

        n_trials = 5 if args.skip_optuna else args.n_trials

        models = {
            "logistic_regression": train_logistic_regression(X_train_bal, y_train_bal),
            "lightgbm": train_lightgbm(
                X_train_bal, y_train_bal, X_val_feat, y_val, n_trials=n_trials
            ),
            "xgboost": train_xgboost(
                X_train_bal, y_train_bal, X_val_feat, y_val, n_trials=n_trials
            ),
        }

        # -- 7. Select champion ------------------------------------------------
        champion_name, champion_model, val_metrics = select_champion(models, X_val_feat, y_val)

        # Log all model metrics
        for model_name, metrics in val_metrics.items():
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)

        # -- 8. Evaluate champion on test set ----------------------------------
        logger.info("Evaluating champion on held-out test set...")
        test_metrics = evaluate_model(champion_model, X_test_feat, y_test, "test")
        mlflow.log_metrics(test_metrics)
        mlflow.log_param("champion_model", champion_name)

        # -- 9. Calibrate intervention thresholds ------------------------------
        logger.info("Calibrating intervention thresholds on validation set...")
        engine = InterventionEngine()
        threshold_result = engine.calibrate_thresholds(
            y_val, champion_model.predict_proba(X_val_feat)[:, 1]
        )
        mlflow.log_params(threshold_result)

        # -- 10. Fairness audit - HARD GATE ------------------------------------
        logger.info("Running fairness audit...")
        val_indices = X_val.index
        fairness_val = fairness_df.loc[fairness_df.index.isin(val_indices)].copy()
        fairness_val["tier"] = engine.predict_batch(
            pd.DataFrame({"vulnerability_score": champion_model.predict_proba(X_val_feat)[:, 1]})
        )["tier"].values

        audit_result = disparate_impact_audit(fairness_val)
        mlflow.log_dict(audit_result, "fairness_audit.json")

        for col, res in audit_result["results"].items():
            mlflow.log_metric(f"fairness_dir_{col}", res["dir"])

        # This will raise ValueError and stop the run if fairness fails
        fairness_gate(audit_result, strict=True)
        logger.info("Fairness gate PASSED.")

        # -- 11. Save artefacts ------------------------------------------------
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        joblib.dump(preprocessor, f"data/processed/{ARTEFACT_PREPROCESSOR}")
        joblib.dump(champion_model, f"data/processed/{ARTEFACT_MODEL}")
        engine.save_thresholds(f"data/processed/{ARTEFACT_THRESHOLDS}")

        # SHAP explainer
        explainer = get_explainer(champion_model, X_train_feat.sample(500, random_state=42))
        joblib.dump(explainer, f"data/processed/{ARTEFACT_SHAP_EXPLAINER}")

        # Save macro baseline for drift monitoring
        save_macro_baseline(macro_snapshot)

        # Log artefacts to MLflow
        mlflow.log_artifacts("data/processed", artifact_path="model")
        mlflow.sklearn.log_model(
            champion_model, "champion_model", registered_model_name="finwatch-champion"
        )

        logger.info("Training complete. Run ID: %s", mlflow.active_run().info.run_id)
        logger.info(
            "Champion: %s | Test PR-AUC: %.4f | Test ROC-AUC: %.4f",
            champion_name,
            test_metrics["test_pr_auc"],
            test_metrics["test_roc_auc"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the FinWatch vulnerability model")
    parser.add_argument("--data-path", default="data/raw/application_train.csv")
    parser.add_argument("--experiment", default=MLFLOW_EXPERIMENT)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--skip-optuna", action="store_true", help="Use n_trials=5 for fast testing"
    )
    args = parser.parse_args()
    main(args)
