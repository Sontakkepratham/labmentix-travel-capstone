"""
Automated ML Pipeline — Travel Flight Type Prediction
Runs end-to-end: load → preprocess → train → evaluate → save model
"""

import os
import sys
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Logging setup ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT
MODELS_DIR = ROOT / "mlops" / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Step 1: Data Loading ───────────────────────────────────────────────────
def load_data():
    log.info("Step 1/6 — Loading data...")
    users   = pd.read_csv(DATA_DIR / "users.csv")
    flights = pd.read_csv(DATA_DIR / "flights.csv")
    hotels  = pd.read_csv(DATA_DIR / "hotels.csv")
    log.info(f"  users={users.shape}, flights={flights.shape}, hotels={hotels.shape}")
    return users, flights, hotels


# ── Step 2: Preprocessing ─────────────────────────────────────────────────
def preprocess(users, flights, hotels):
    log.info("Step 2/6 — Preprocessing...")

    flights["date"] = pd.to_datetime(flights["date"], format="%m/%d/%Y")
    hotels["date"]  = pd.to_datetime(hotels["date"],  format="%m/%d/%Y")

    flights["month"] = flights["date"].dt.month
    flights["year"]  = flights["date"].dt.year

    # One record per trip
    df = flights.drop_duplicates(subset="travelCode", keep="first").copy()

    # Merge user demographics
    df = df.merge(users, left_on="userCode", right_on="code", how="left")
    df.rename(columns={"name_x": "route", "name_y": "user_name"}, inplace=True)

    # Hotel features per trip
    hotel_feats = hotels.groupby("travelCode").agg(
        hotel_days=("days", "sum"),
        hotel_total=("total", "sum"),
    ).reset_index()
    df = df.merge(hotel_feats, on="travelCode", how="left")
    df["hotel_days"]  = df["hotel_days"].fillna(0)
    df["hotel_total"] = df["hotel_total"].fillna(0)
    df["hotel_booked"] = (df["hotel_days"] > 0).astype(int)

    # Derived features
    df["price_per_km"] = df["price"] / df["distance"]
    df["speed_kmh"]    = df["distance"] / df["time"]
    bins, lbl = [20, 30, 40, 50, 65], [0, 1, 2, 3]
    df["age_group"] = pd.cut(df["age"], bins=bins, labels=lbl).astype(float).fillna(1)

    # Winsorize hotel_total
    Q1, Q3 = df["hotel_total"].quantile(0.25), df["hotel_total"].quantile(0.75)
    df["hotel_total"] = df["hotel_total"].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

    # Log transforms
    df["price_log"]       = np.log1p(df["price"])
    df["hotel_total_log"] = np.log1p(df["hotel_total"])

    log.info(f"  Dataset after preprocessing: {df.shape}")
    return df


# ── Step 3: Feature Engineering & Encoding ────────────────────────────────
FEATURE_COLS = [
    "price", "distance", "time", "age", "age_group",
    "hotel_days", "hotel_total", "hotel_booked", "month",
    "agency", "gender", "company", "from", "to",
    "price_log", "hotel_total_log",
]
CAT_COLS   = ["agency", "gender", "company", "from", "to"]
TARGET_COL = "flightType"


def encode(df):
    log.info("Step 3/6 — Encoding features...")

    le_target = LabelEncoder()
    y = le_target.fit_transform(df[TARGET_COL])

    X = df[FEATURE_COLS].copy()
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    log.info(f"  Features={X_scaled.shape[1]}, Samples={X_scaled.shape[0]}")
    return X_scaled, y, le_target, encoders, scaler


# ── Step 4: Train/Test Split + SMOTE ──────────────────────────────────────
def split_and_balance(X, y):
    log.info("Step 4/6 — Splitting and balancing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    log.info(f"  Train={X_train_bal.shape[0]}, Test={X_test.shape[0]}")
    return X_train_bal, X_test, y_train_bal, y_test


# ── Step 5: Train & Evaluate Models ───────────────────────────────────────
MODELS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, random_state=42, solver="lbfgs"
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    ),
}


def train_and_evaluate(X_train, X_test, y_train, y_test, le_target):
    log.info("Step 5/6 — Training models...")
    results = {}

    for name, model in MODELS.items():
        log.info(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "f1":        round(f1_score(y_test, y_pred, average="weighted"), 4),
            "precision": round(precision_score(y_test, y_pred, average="weighted"), 4),
            "recall":    round(recall_score(y_test, y_pred, average="weighted"), 4),
        }
        results[name] = {"model": model, "metrics": metrics}
        log.info(f"    Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}")

    # Pick best model by F1
    best_name = max(results, key=lambda n: results[n]["metrics"]["f1"])
    log.info(f"  Best model: {best_name} (F1={results[best_name]['metrics']['f1']})")
    return results, best_name


# ── Step 6: Save Artifacts ────────────────────────────────────────────────
def save_artifacts(results, best_name, le_target, encoders, scaler):
    log.info("Step 6/6 — Saving artifacts...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    artifacts = {
        "best_model_name": best_name,
        "best_model":      results[best_name]["model"],
        "le_target":       le_target,
        "encoders":        encoders,
        "scaler":          scaler,
        "feature_cols":    FEATURE_COLS,
        "cat_cols":        CAT_COLS,
        "classes":         list(le_target.classes_),
    }

    model_path = MODELS_DIR / f"model_{ts}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(artifacts, f)

    # Also save as "latest" for the API to always pick up
    latest_path = MODELS_DIR / "model_latest.pkl"
    with open(latest_path, "wb") as f:
        pickle.dump(artifacts, f)

    # Save metrics JSON
    metrics_all = {n: r["metrics"] for n, r in results.items()}
    metrics_path = MODELS_DIR / f"metrics_{ts}.json"
    with open(metrics_path, "w") as f:
        json.dump({"timestamp": ts, "models": metrics_all, "best": best_name}, f, indent=2)

    log.info(f"  Model saved    : {model_path}")
    log.info(f"  Latest model   : {latest_path}")
    log.info(f"  Metrics saved  : {metrics_path}")
    return str(latest_path)


# ── Main Pipeline ──────────────────────────────────────────────────────────
def run_pipeline():
    log.info("=" * 55)
    log.info("  TRAVEL FLIGHT TYPE PREDICTION — ML PIPELINE")
    log.info("=" * 55)
    start = datetime.now()

    users, flights, hotels = load_data()
    df                     = preprocess(users, flights, hotels)
    X, y, le_target, encoders, scaler = encode(df)
    X_train, X_test, y_train, y_test  = split_and_balance(X, y)
    results, best_name                 = train_and_evaluate(X_train, X_test, y_train, y_test, le_target)
    model_path                         = save_artifacts(results, best_name, le_target, encoders, scaler)

    elapsed = (datetime.now() - start).seconds
    log.info(f"\nPipeline complete in {elapsed}s — best model: {best_name}")
    log.info(f"Metrics: {results[best_name]['metrics']}")
    return model_path


if __name__ == "__main__":
    run_pipeline()
