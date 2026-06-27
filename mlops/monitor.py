"""
Streamlit Monitoring Dashboard — Travel Flight Type Prediction

Run:
    streamlit run monitor.py

Shows:
  - Model performance metrics across all training runs
  - Live prediction tester
  - Data drift detection (feature distribution changes)
  - MLflow run comparison
"""

import json
import pickle
import glob
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests

# ── Config ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR   = ROOT
API_URL    = "http://localhost:8000"

st.set_page_config(
    page_title="Travel ML Monitor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("✈️ Travel ML Monitor")
page = st.sidebar.radio(
    "Navigation",
    ["📊 Model Performance", "🔮 Live Predictor", "📈 Data Drift", "📋 Run History"],
)

# ── Helpers ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_latest_model():
    path = MODELS_DIR / "model_latest.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metrics_history():
    files = sorted(glob.glob(str(MODELS_DIR / "metrics_*.json")))
    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        ts = d.get("timestamp", "unknown")
        best = d.get("best", "")
        for model_name, metrics in d.get("models", {}).items():
            rows.append({"timestamp": ts, "model": model_name, "best": model_name == best, **metrics})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


@st.cache_data
def load_data():
    users   = pd.read_csv(DATA_DIR / "users.csv")
    flights = pd.read_csv(DATA_DIR / "flights.csv")
    hotels  = pd.read_csv(DATA_DIR / "hotels.csv")
    flights["date"] = pd.to_datetime(flights["date"], format="%m/%d/%Y")
    return users, flights, hotels


# ── Page 1: Model Performance ───────────────────────────────────────────────
if page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")

    artifacts = load_latest_model()
    if artifacts is None:
        st.warning("⚠️ No trained model found. Run `pipeline.py` or `train.py` first.")
        st.code("cd mlops\npython pipeline.py")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Model", artifacts["best_model_name"])
    col2.metric("Classes", ", ".join(artifacts["classes"]))
    col3.metric("Features", len(artifacts["feature_cols"]))

    st.divider()
    hist_df = load_metrics_history()

    if not hist_df.empty:
        st.subheader("Latest Training Run — All Models")
        latest_ts = hist_df["timestamp"].max()
        latest = hist_df[hist_df["timestamp"] == latest_ts].copy()

        metric_cols = ["accuracy", "f1", "precision", "recall"]
        st.dataframe(
            latest[["model"] + metric_cols].style
            .highlight_max(subset=metric_cols, color="#d4edda")
            .format({c: "{:.4f}" for c in metric_cols}),
            use_container_width=True,
        )

        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(latest))
        w = 0.2
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
        for i, m in enumerate(metric_cols):
            ax.bar(x + i*w, latest[m].values, w, label=m, color=colors[i], alpha=0.85)
        ax.set_xticks(x + w*1.5)
        ax.set_xticklabels(latest["model"], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title("Model Comparison — Latest Run", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

        # History trend
        if hist_df["timestamp"].nunique() > 1:
            st.subheader("F1 Score Trend Across Runs")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            for model_name in hist_df["model"].unique():
                sub = hist_df[hist_df["model"] == model_name].sort_values("timestamp")
                ax2.plot(sub["timestamp"], sub["f1"], marker="o", label=model_name)
            ax2.set_title("F1 Score Over Time by Model", fontweight="bold")
            ax2.set_ylabel("F1 Score (weighted)")
            ax2.legend()
            ax2.grid(alpha=0.3)
            plt.xticks(rotation=30)
            st.pyplot(fig2)
    else:
        st.info("No metrics history yet. Run the pipeline to generate metrics.")

    st.subheader("Feature Importance")
    model = artifacts["best_model"]
    feat_cols = artifacts["feature_cols"]
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=True)
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        imp.plot(kind="barh", ax=ax3, color="#2196F3", alpha=0.85, edgecolor="black")
        ax3.set_title(f"Feature Importance — {artifacts['best_model_name']}", fontweight="bold")
        st.pyplot(fig3)


# ── Page 2: Live Predictor ──────────────────────────────────────────────────
elif page == "🔮 Live Predictor":
    st.title("🔮 Live Flight Type Predictor")
    st.info("Fill in trip details to get a real-time prediction from the API.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Trip Details")
        price    = st.number_input("Ticket Price (R$)", 100.0, 3000.0, 1200.0, 50.0)
        distance = st.number_input("Distance (km)", 100.0, 2000.0, 676.5, 10.0)
        time     = st.number_input("Flight Duration (hours)", 0.3, 5.0, 1.76, 0.1)
        agency   = st.selectbox("Agency", ["FlyingDrops", "CloudFy", "Rainbow"])
        origin   = st.selectbox("Origin", [
            "Recife (PE)", "Brasilia (DF)", "Aracaju (SE)",
            "Sao Paulo (SP)", "Salvador (BH)", "Natal (RN)",
            "Campo Grande (MS)", "Florianopolis (SC)",
        ])
        dest     = st.selectbox("Destination", [
            "Florianopolis (SC)", "Recife (PE)", "Salvador (BH)",
            "Campo Grande (MS)", "Sao Paulo (SP)", "Natal (RN)",
            "Brasilia (DF)", "Aracaju (SE)",
        ])
        month    = st.slider("Month of Travel", 1, 12, 9)

    with col2:
        st.subheader("Passenger Details")
        age      = st.slider("Age", 21, 65, 35)
        gender   = st.selectbox("Gender", ["male", "female", "none"])
        company  = st.selectbox("Company", [
            "4You", "Monsters CYA", "Wonka Company", "Acme Factory", "Umbrella LTDA"
        ])
        hotel    = st.checkbox("Hotel Booked?")
        hotel_days  = st.number_input("Hotel Days", 0, 14, 3) if hotel else 0
        hotel_total = st.number_input("Hotel Total (R$)", 0.0, 5000.0, 900.0) if hotel else 0.0

    st.divider()
    if st.button("🚀 Predict Flight Class", type="primary", use_container_width=True):
        payload = {
            "price": price, "distance": distance, "time": time,
            "age": age, "hotel_days": hotel_days, "hotel_total": hotel_total,
            "hotel_booked": int(hotel), "month": month,
            "agency": agency, "gender": gender, "company": company,
            "origin": origin, "destination": dest,
        }
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                pred   = result["predicted_class"]
                conf   = result["confidence"]
                proba  = result["probabilities"]

                color_map = {"economic": "🟢", "premium": "🟡", "firstClass": "🔵"}
                st.success(f"## {color_map.get(pred, '⚪')} Predicted: **{pred.upper()}**  (confidence: {conf*100:.1f}%)")

                cols = st.columns(3)
                for i, (cls, prob) in enumerate(proba.items()):
                    cols[i].metric(cls.capitalize(), f"{prob*100:.1f}%")

                # Probability bar
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ["#4CAF50", "#FF9800", "#2196F3"]
                ax.barh(list(proba.keys()), list(proba.values()), color=colors, alpha=0.85)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title("Class Probabilities", fontweight="bold")
                st.pyplot(fig)
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Start it with:\n```\nuvicorn app:app --reload --port 8000\n```")

    # Direct model prediction fallback
    st.divider()
    with st.expander("🔧 Direct model prediction (no API needed)"):
        artifacts = load_latest_model()
        if artifacts and st.button("Predict directly from model"):
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from app import build_features, TripInput
            trip = TripInput(
                price=price, distance=distance, time=time, age=age,
                hotel_days=hotel_days, hotel_total=hotel_total,
                hotel_booked=int(hotel), month=month,
                agency=agency, gender=gender, company=company,
                origin=origin, destination=dest,
            )
            # Temporarily set _artifacts
            import app as app_module
            app_module._artifacts = artifacts
            X = build_features(trip)
            model = artifacts["best_model"]
            pred  = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            classes = artifacts["classes"]
            st.success(f"Predicted: **{classes[pred]}** (confidence: {proba[pred]*100:.1f}%)")


# ── Page 3: Data Drift ──────────────────────────────────────────────────────
elif page == "📈 Data Drift":
    st.title("📈 Data Distribution & Drift Detection")
    st.info("Compare feature distributions over time to detect data drift.")

    try:
        users, flights, hotels = load_data()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

    st.subheader("Flight Bookings Over Time")
    monthly = flights.groupby(flights["date"].dt.to_period("M")).size()
    monthly.index = monthly.index.astype(str)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly.index, monthly.values, marker="o", linewidth=2, color="#2196F3")
    ax.fill_between(monthly.index, monthly.values, alpha=0.2, color="#2196F3")
    ax.set_title("Monthly Booking Volume", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Bookings")
    plt.xticks(rotation=45)
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Feature Distribution by Year (Drift Check)")
    col1, col2 = st.columns(2)

    with col1:
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        for yr in sorted(flights["year"].unique()):
            sub = flights[flights["year"] == yr]["price"]
            sub.plot.kde(ax=ax2, label=str(yr), linewidth=1.5)
        ax2.set_title("Price Distribution by Year", fontweight="bold")
        ax2.set_xlabel("Price (R$)")
        ax2.legend(title="Year")
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        ft_year = flights.groupby(["year", "flightType"]).size().unstack(fill_value=0)
        ft_year_pct = ft_year.div(ft_year.sum(axis=1), axis=0)
        ft_year_pct.plot(kind="bar", ax=ax3, colormap="Set2", alpha=0.85, edgecolor="white")
        ax3.set_title("Flight Type Share by Year", fontweight="bold")
        ax3.set_ylabel("Proportion")
        ax3.tick_params(axis="x", rotation=0)
        ax3.legend(title="Flight Type")
        st.pyplot(fig3)

    st.subheader("Statistical Drift Test (KS Test — Price)")
    years = sorted(flights["year"].unique())
    if len(years) >= 2:
        from scipy import stats
        ref_year = st.selectbox("Reference year", years[:-1])
        cmp_year = st.selectbox("Compare year", [y for y in years if y != ref_year], index=len(years)-2)
        ref_data = flights[flights["year"] == ref_year]["price"]
        cmp_data = flights[flights["year"] == cmp_year]["price"]
        ks_stat, ks_p = stats.ks_2samp(ref_data, cmp_data)
        col1, col2, col3 = st.columns(3)
        col1.metric("KS Statistic", f"{ks_stat:.4f}")
        col2.metric("P-Value", f"{ks_p:.4f}")
        col3.metric("Drift Detected?", "Yes ⚠️" if ks_p < 0.05 else "No ✅")
        if ks_p < 0.05:
            st.warning(f"Significant price distribution shift detected between {ref_year} and {cmp_year}.")
        else:
            st.success(f"No significant drift detected between {ref_year} and {cmp_year}.")


# ── Page 4: Run History ─────────────────────────────────────────────────────
elif page == "📋 Run History":
    st.title("📋 Training Run History")

    hist_df = load_metrics_history()
    if hist_df.empty:
        st.info("No run history yet. Train a model first.")
        st.code("cd mlops\npython pipeline.py")
    else:
        st.dataframe(hist_df.style.format({
            "accuracy": "{:.4f}", "f1": "{:.4f}",
            "precision": "{:.4f}", "recall": "{:.4f}",
        }), use_container_width=True)

        csv = hist_df.to_csv(index=False)
        st.download_button("⬇️ Download Run History CSV", csv, "run_history.csv", "text/csv")

    st.divider()
    st.subheader("MLflow UI")
    st.info("For detailed run tracking with MLflow, run:")
    st.code("cd mlops\nmlflow ui\n# Then open http://localhost:5000")
