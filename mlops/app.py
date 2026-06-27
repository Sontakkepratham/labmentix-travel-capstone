"""
FastAPI Model Serving — Travel Flight Type Prediction

Run:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /             → health check
    GET  /model/info   → loaded model info
    POST /predict      → single prediction
    POST /predict/batch → batch predictions
"""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Travel Flight Type Prediction API",
    description="Predicts flight class (economic / premium / firstClass) based on trip and user features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ─────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "models" / "model_latest.pkl"
_artifacts = None


def load_model():
    global _artifacts
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run pipeline.py or train.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        _artifacts = pickle.load(f)
    log.info(f"Model loaded: {_artifacts['best_model_name']}")


@app.on_event("startup")
def startup():
    load_model()


# ── Request / Response schemas ─────────────────────────────────────────────
class TripInput(BaseModel):
    price: float           = Field(..., example=1200.0,  description="Ticket price in R$")
    distance: float        = Field(..., example=676.5,   description="Route distance in km")
    time: float            = Field(..., example=1.76,    description="Flight duration in hours")
    age: int               = Field(..., example=35,      description="Passenger age")
    hotel_days: int        = Field(0,   example=3,       description="Hotel nights booked (0 if none)")
    hotel_total: float     = Field(0.0, example=900.0,   description="Total hotel spend in R$")
    hotel_booked: int      = Field(0,   example=1,       description="1 if hotel booked, 0 otherwise")
    month: int             = Field(..., example=9,       description="Month of travel (1-12)")
    agency: str            = Field(..., example="FlyingDrops", description="Booking agency")
    gender: str            = Field(..., example="male",  description="Passenger gender")
    company: str           = Field(..., example="4You",  description="Passenger's company")
    origin: str            = Field(..., example="Recife (PE)",        description="Origin city")
    destination: str       = Field(..., example="Florianopolis (SC)", description="Destination city")

    @validator("month")
    def valid_month(cls, v):
        if not 1 <= v <= 12:
            raise ValueError("month must be between 1 and 12")
        return v

    @validator("age")
    def valid_age(cls, v):
        if not 18 <= v <= 100:
            raise ValueError("age must be between 18 and 100")
        return v


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict
    model_used: str


class BatchInput(BaseModel):
    trips: List[TripInput]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int


# ── Feature builder ────────────────────────────────────────────────────────
def build_features(trip: TripInput) -> pd.DataFrame:
    a = _artifacts
    bins, lbl = [20, 30, 40, 50, 65], [0, 1, 2, 3]
    age_group = pd.cut([trip.age], bins=bins, labels=lbl)[0]
    age_group = float(age_group) if not pd.isna(age_group) else 1.0

    row = {
        "price":           trip.price,
        "distance":        trip.distance,
        "time":            trip.time,
        "age":             trip.age,
        "age_group":       age_group,
        "hotel_days":      trip.hotel_days,
        "hotel_total":     trip.hotel_total,
        "hotel_booked":    trip.hotel_booked,
        "month":           trip.month,
        "agency":          trip.agency,
        "gender":          trip.gender,
        "company":         trip.company,
        "from":            trip.origin,
        "to":              trip.destination,
        "price_log":       np.log1p(trip.price),
        "hotel_total_log": np.log1p(trip.hotel_total),
    }

    df = pd.DataFrame([row])

    # Encode categoricals — handle unseen labels gracefully
    cat_cols = ["agency", "gender", "company", "from", "to"]
    for col in cat_cols:
        le = a["encoders"][col]
        val = str(df[col].iloc[0])
        if val in le.classes_:
            df[col] = le.transform([val])
        else:
            df[col] = 0  # unknown → 0

    # Ensure column order matches training
    feature_cols = a["feature_cols"]
    df = df[feature_cols]

    # Scale
    df_scaled = a["scaler"].transform(df)
    return pd.DataFrame(df_scaled, columns=feature_cols)


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health():
    return {
        "status": "ok",
        "service": "Travel Flight Type Prediction API",
        "model": _artifacts["best_model_name"] if _artifacts else "not loaded",
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    if not _artifacts:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_name": _artifacts["best_model_name"],
        "classes":    _artifacts["classes"],
        "n_features": len(_artifacts["feature_cols"]),
        "features":   _artifacts["feature_cols"],
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(trip: TripInput):
    if not _artifacts:
        raise HTTPException(503, "Model not loaded")
    try:
        X = build_features(trip)
        model = _artifacts["best_model"]
        pred  = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = _artifacts["classes"]

        return PredictionResponse(
            predicted_class=classes[pred],
            confidence=round(float(proba[pred]), 4),
            probabilities={cls: round(float(p), 4) for cls, p in zip(classes, proba)},
            model_used=_artifacts["best_model_name"],
        )
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(500, str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    if not _artifacts:
        raise HTTPException(503, "Model not loaded")
    predictions = []
    for trip in batch.trips:
        try:
            X = build_features(trip)
            model = _artifacts["best_model"]
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            classes = _artifacts["classes"]
            predictions.append(PredictionResponse(
                predicted_class=classes[pred],
                confidence=round(float(proba[pred]), 4),
                probabilities={cls: round(float(p), 4) for cls, p in zip(classes, proba)},
                model_used=_artifacts["best_model_name"],
            ))
        except Exception as e:
            log.error(f"Batch item error: {e}")
            predictions.append(PredictionResponse(
                predicted_class="error",
                confidence=0.0,
                probabilities={},
                model_used="error",
            ))
    return BatchResponse(predictions=predictions, total=len(predictions))


@app.post("/model/reload", tags=["Model"])
def reload_model():
    """Hot-reload the latest model without restarting the server."""
    try:
        load_model()
        return {"status": "reloaded", "model": _artifacts["best_model_name"]}
    except Exception as e:
        raise HTTPException(500, str(e))
