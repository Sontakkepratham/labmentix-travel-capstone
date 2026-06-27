"""
MLflow Experiment Tracking — Travel Flight Type Prediction
Trains all 3 models with full MLflow logging: params, metrics, artifacts.

Run:
    python train.py

Then view results:
    mlflow ui
    → open http://localhost:5000
"""

import pickle
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier

# Import shared preprocessing from pipeline
sys_path = str(Path(__file__).parent)
import sys
sys.path.insert(0, sys_path)
from pipeline import load_data, preprocess, encode, split_and_balance

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).parent.parent
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── MLflow setup ───────────────────────────────────────────────────────────
EXPERIMENT_NAME = "Travel_FlightType_Prediction"
mlflow.set_tracking_uri("mlruns")           # stores locally in ./mlruns/
mlflow.set_experiment(EXPERIMENT_NAME)

# ── Model configs ──────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=1000, random_state=42,
                                     multi_class="multinomial", solver="lbfgs"),
        "params": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000},
        "log_fn": mlflow.sklearn.log_model,
        "artifact_path": "logistic_regression",
    },
    {
        "name": "Random Forest",
        "model": RandomForestClassifier(n_estimators=200, max_depth=20,
                                         random_state=42, n_jobs=-1),
        "params": {"n_estimators": 200, "max_depth": 20, "random_state": 42},
        "log_fn": mlflow.sklearn.log_model,
        "artifact_path": "random_forest",
    },
    {
        "name": "XGBoost",
        "model": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                                eval_metric="mlogloss", use_label_encoder=False,
                                random_state=42, n_jobs=-1),
        "params": {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
        "log_fn": mlflow.xgboost.log_model,
        "artifact_path": "xgboost",
    },
]


def run_training():
    # ── Data prep ────────────────────────────────────────────────────────
    print("Loading and preprocessing data...")
    users, flights, hotels = load_data()
    df = preprocess(users, flights, hotels)
    X, y, le_target, encoders, scaler = encode(df)
    X_train, X_test, y_train, y_test = split_and_balance(X, y)
    classes = list(le_target.classes_)

    print(f"Classes: {classes}")
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}\n")

    best_run_id  = None
    best_f1      = 0.0
    best_model   = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for cfg in MODEL_CONFIGS:
        print(f"{'─'*50}")
        print(f"Training: {cfg['name']}")

        with mlflow.start_run(run_name=cfg["name"]) as run:
            run_id = run.info.run_id

            # ── Log parameters ─────────────────────────────────────────
            mlflow.log_params(cfg["params"])
            mlflow.log_param("smote", True)
            mlflow.log_param("train_size", X_train.shape[0])
            mlflow.log_param("test_size",  X_test.shape[0])
            mlflow.log_param("n_features",  X_train.shape[1])

            # ── Train ──────────────────────────────────────────────────
            model = cfg["model"]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # ── Metrics ────────────────────────────────────────────────
            acc  = accuracy_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted")
            rec  = recall_score(y_test, y_pred, average="weighted")

            mlflow.log_metric("accuracy",  acc)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.log_metric("precision_weighted", prec)
            mlflow.log_metric("recall_weighted", rec)

            # Per-class metrics
            report = classification_report(y_test, y_pred,
                                           target_names=classes, output_dict=True)
            for cls in classes:
                mlflow.log_metric(f"f1_{cls}",        report[cls]["f1-score"])
                mlflow.log_metric(f"precision_{cls}",  report[cls]["precision"])
                mlflow.log_metric(f"recall_{cls}",     report[cls]["recall"])

            # ── Cross-validation ───────────────────────────────────────
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                         scoring="f1_weighted", n_jobs=-1)
            mlflow.log_metric("cv_f1_mean", cv_scores.mean())
            mlflow.log_metric("cv_f1_std",  cv_scores.std())

            # ── Log model ──────────────────────────────────────────────
            if cfg["name"] == "XGBoost":
                mlflow.xgboost.log_model(model, cfg["artifact_path"])
            else:
                mlflow.sklearn.log_model(model, cfg["artifact_path"])

            # ── Tags ───────────────────────────────────────────────────
            mlflow.set_tag("model_type", cfg["name"])
            mlflow.set_tag("dataset", "travel_capstone")
            mlflow.set_tag("target", "flightType")

            print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  CV-F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
            print(f"  Run ID: {run_id}")

            if f1 > best_f1:
                best_f1    = f1
                best_run_id = run_id
                best_model  = model
                best_name   = cfg["name"]

    # ── Save best model as production artifact ─────────────────────────
    print(f"\n{'='*50}")
    print(f"Best model: {best_name}  (F1={best_f1:.4f})")
    print(f"Best run ID: {best_run_id}")

    artifacts = {
        "best_model_name": best_name,
        "best_model":      best_model,
        "le_target":       le_target,
        "encoders":        encoders,
        "scaler":          scaler,
        "feature_cols":    list(X.columns),
        "classes":         classes,
    }
    latest_path = MODELS_DIR / "model_latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Production model saved → {latest_path}")
    print(f"\nView MLflow UI: run 'mlflow ui' then open http://localhost:5000")
    return str(latest_path)


if __name__ == "__main__":
    run_training()
