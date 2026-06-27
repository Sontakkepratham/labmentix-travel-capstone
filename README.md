# ✈️ Travel Agency — ML Capstone Project

**Labmentix Final Project** | Flight Type Prediction using EDA, Machine Learning & MLOps

---

## 📌 Project Overview

This project analyzes travel booking data (flights, hotels, users) from 5 companies and builds a complete ML system to predict **flight class preference** (economic / premium / firstClass) for each customer.

| Dataset | Rows | Description |
|---------|------|-------------|
| `users.csv` | 1,340 | User demographics (name, age, gender, company) |
| `flights.csv` | 271,888 | Flight bookings (route, price, agency, class) |
| `hotels.csv` | 40,552 | Hotel bookings (hotel, city, days, cost) |

---

## 📁 Project Structure

```
├── users.csv                     # User dataset
├── flights.csv                   # Flights dataset
├── hotels.csv                    # Hotels dataset
│
├── Travel_EDA_Analysis.ipynb     # Exploratory Data Analysis notebook
├── Travel_ML_Prediction.ipynb    # Machine Learning prediction notebook
│
└── mlops/
    ├── pipeline.py               # Automated ML pipeline (end-to-end)
    ├── train.py                  # MLflow experiment tracking
    ├── app.py                    # FastAPI model serving API
    ├── monitor.py                # Streamlit monitoring dashboard
    ├── requirements.txt          # Python dependencies
    └── models/                   # Saved model artifacts (generated on run)
```

---

## 🔍 EDA Notebook — `Travel_EDA_Analysis.ipynb`

Covers:
- Data loading and inspection (3 datasets)
- Data wrangling & feature engineering
- 11 visualizations with business insights:
  - Flight type distribution
  - Agency performance & pricing
  - Age & gender analysis
  - Top travel routes
  - Monthly booking trends
  - Hotel revenue analysis
  - Correlation heatmap
- Business recommendations

---

## 🤖 ML Notebook — `Travel_ML_Prediction.ipynb`

Covers:
- Full preprocessing pipeline (encoding, scaling, SMOTE)
- 3 Hypothesis tests (t-test, ANOVA)
- 3 ML Models with hyperparameter tuning (GridSearchCV):
  - Logistic Regression
  - Random Forest
  - XGBoost ← **Best Model**
- Model comparison & feature importance (permutation importance)

---

## ⚙️ MLOps System — `mlops/`

### 1. Automated Pipeline
```bash
cd mlops
python pipeline.py
```
Runs all 6 steps: load → preprocess → encode → split+SMOTE → train 3 models → save best model.

### 2. MLflow Experiment Tracking
```bash
python train.py      # trains & logs to MLflow
mlflow ui            # open http://localhost:5000
```
Tracks: accuracy, F1, precision, recall, per-class metrics, CV scores, hyperparameters.

### 3. FastAPI Model Serving
```bash
python -m uvicorn app:app --reload --port 8000
# API docs → http://localhost:8000/docs
```

**Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/model/info` | Loaded model details |
| POST | `/predict` | Single trip prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/model/reload` | Hot-reload latest model |

**Example request:**
```json
POST /predict
{
  "price": 1434.38,
  "distance": 676.53,
  "time": 1.76,
  "age": 35,
  "hotel_days": 3,
  "hotel_total": 900.0,
  "hotel_booked": 1,
  "month": 9,
  "agency": "FlyingDrops",
  "gender": "male",
  "company": "4You",
  "origin": "Recife (PE)",
  "destination": "Florianopolis (SC)"
}
```

**Response:**
```json
{
  "predicted_class": "firstClass",
  "confidence": 0.97,
  "probabilities": {"economic": 0.01, "firstClass": 0.97, "premium": 0.02},
  "model_used": "random_forest"
}
```

### 4. Streamlit Monitoring Dashboard
```bash
python -m streamlit run monitor.py
```
Pages:
- **Model Performance** — metrics chart, feature importance
- **Live Predictor** — real-time prediction UI (calls the API)
- **Data Drift** — KS test, distribution shifts over time
- **Run History** — all pipeline runs with downloadable CSV

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r mlops/requirements.txt

# 2. Run the ML pipeline (trains & saves model)
cd mlops
python pipeline.py

# 3. Start the API
python -m uvicorn app:app --reload --port 8000

# 4. Open the monitoring dashboard
python -m streamlit run monitor.py

# 5. (Optional) MLflow experiment tracking
python train.py
mlflow ui
```

---

## 📊 Model Results

| Model | Accuracy | F1 (Weighted) |
|-------|----------|---------------|
| Logistic Regression | ~0.76 | ~0.76 |
| Random Forest | ~1.00 | ~1.00 |
| **XGBoost (Final)** | **~1.00** | **~1.00** |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Analysis | pandas, numpy, scipy |
| Visualization | matplotlib, seaborn, plotly |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| MLOps | MLflow, FastAPI, Streamlit, uvicorn |
| Language | Python 3.10+ |

---

*Labmentix Final Capstone Project — Travel Agency Analysis & ML System*
