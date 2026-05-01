import os
import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import time

from datetime import datetime, timedelta
from pytz import timezone
from tensorflow.keras.models import load_model

from priority_and_dominance import analyze_priority_and_dominance
from knowledge_engine import update_24hr_knowledge_graph, extract_graph_for_dashboard
from explain_prediction import generate_explanation_report


# =========================================================
# CACHE CONFIG (NEW)
# =========================================================

CACHE_TTL = 3600  # 1 hour
_cache = {}


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_LAT = 28.6562
DEFAULT_LON = 77.2300
TIMEZONE = "Asia/Kolkata"

SEQ_LEN = 24
FORECAST_HOURS = 72

MODELS_DIR = "models"

ENSEMBLE_WEIGHTS = {
    "LSTM": 0.55,
    "BiLSTM": 0.25,
    "GRU": 0.20
}

TRAINED_CITY_COLUMNS = [
    "City_Pune",
    "City_Shillong",
    "City_Vashi"
]


# =========================================================
# FAST MODEL LOADING
# =========================================================

tf.config.run_functions_eagerly(False)

scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.pkl"))

lstm_model = load_model(os.path.join(MODELS_DIR, "lstm_residual.keras"))
bilstm_model = load_model(os.path.join(MODELS_DIR, "bilstm_residual.keras"))
gru_model = load_model(os.path.join(MODELS_DIR, "gru_residual.keras"))

predict_lstm = tf.function(lstm_model)
predict_bilstm = tf.function(bilstm_model)
predict_gru = tf.function(gru_model)


# =========================================================
# CPCB SUBINDEX FUNCTIONS
# =========================================================

def linear_subindex(c, bp_lo, bp_hi, i_lo, i_hi):
    return ((i_hi - i_lo) / (bp_hi - bp_lo)) * (c - bp_lo) + i_lo


def subindex_pm25(c):
    if c <= 30: return linear_subindex(c, 0, 30, 0, 50)
    if c <= 60: return linear_subindex(c, 31, 60, 51, 100)
    if c <= 90: return linear_subindex(c, 61, 90, 101, 200)
    if c <= 120: return linear_subindex(c, 91, 120, 201, 300)
    if c <= 250: return linear_subindex(c, 121, 250, 301, 400)
    return linear_subindex(c, 251, 500, 401, 500)


def subindex_pm10(c):
    if c <= 50: return linear_subindex(c, 0, 50, 0, 50)
    if c <= 100: return linear_subindex(c, 51, 100, 51, 100)
    if c <= 250: return linear_subindex(c, 101, 250, 101, 200)
    if c <= 350: return linear_subindex(c, 251, 350, 201, 300)
    if c <= 430: return linear_subindex(c, 351, 430, 301, 400)
    return linear_subindex(c, 431, 600, 401, 500)


def subindex_no2(c):
    if c <= 40: return linear_subindex(c, 0, 40, 0, 50)
    if c <= 80: return linear_subindex(c, 41, 80, 51, 100)
    if c <= 180: return linear_subindex(c, 81, 180, 101, 200)
    if c <= 280: return linear_subindex(c, 181, 280, 201, 300)
    if c <= 400: return linear_subindex(c, 281, 400, 301, 400)
    return linear_subindex(c, 401, 1000, 401, 500)


def subindex_o3(c):
    if c <= 50: return linear_subindex(c, 0, 50, 0, 50)
    if c <= 100: return linear_subindex(c, 51, 100, 51, 100)
    if c <= 168: return linear_subindex(c, 101, 168, 101, 200)
    if c <= 208: return linear_subindex(c, 169, 208, 201, 300)
    if c <= 748: return linear_subindex(c, 209, 748, 301, 400)
    return linear_subindex(c, 749, 1000, 401, 500)


def subindex_co(c):
    if c <= 1: return linear_subindex(c, 0, 1, 0, 50)
    if c <= 2: return linear_subindex(c, 1.1, 2, 51, 100)
    if c <= 10: return linear_subindex(c, 2.1, 10, 101, 200)
    if c <= 17: return linear_subindex(c, 10.1, 17, 201, 300)
    if c <= 34: return linear_subindex(c, 17.1, 34, 301, 400)
    return linear_subindex(c, 34.1, 50, 401, 500)


# =========================================================
# DATA FETCH
# =========================================================

def fetch_air_quality_history(start_date, end_date, lat, lon):

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["pm2_5","pm10","nitrogen_dioxide","ozone","carbon_monoxide"],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.set_index("time", inplace=True)

    df.rename(columns={
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "nitrogen_dioxide": "NO2",
        "ozone": "Ozone",
        "carbon_monoxide": "CO_ug"
    }, inplace=True)

    df["CO"] = df["CO_ug"] / 1000
    df.drop(columns=["CO_ug"], inplace=True)

    return df


def fetch_weather(start_date, end_date, lat, lon, forecast=False):

    url = (
        "https://api.open-meteo.com/v1/forecast"
        if forecast else
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "pressure_msl",
            "windspeed_10m",
            "winddirection_10m"
        ]),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df.set_index("time")


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_aqi_pipeline(lat=DEFAULT_LAT, lon=DEFAULT_LON):

    lat = float(lat)
    lon = float(lon)

    key = (round(lat, 4), round(lon, 4))
    now_ts = time.time()

    # CACHE CHECK
    if key in _cache:
        entry = _cache[key]
        age = now_ts - entry["ts"]

        print(f"CACHE AGE: {age:.2f} seconds")

        if age < CACHE_TTL:
            print("✅ CACHE HIT (fast response)")
            return entry["data"]
        else:
            print("❌ CACHE EXPIRED (recomputing...)")
    else:
        print("🧠 CACHE MISS (first computation)")

    ist = timezone(TIMEZONE)
    now = datetime.now(ist)

    hist_start = (now - timedelta(days=2)).date()
    hist_end = now.date()

    aq_hist = fetch_air_quality_history(hist_start, hist_end, lat, lon)
    wx_hist = fetch_weather(hist_start, hist_end, lat, lon)

    df = aq_hist.join(wx_hist).dropna()
    df.index = df.index.tz_localize(TIMEZONE)

    end_time = now.replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(hours=24)

    df = df[(df.index >= start_time) & (df.index < end_time)]
    df = df.tail(SEQ_LEN)

    if len(df) != SEQ_LEN:
        raise RuntimeError(f"Expected 24 rows, got {len(df)}")

    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    latest = df.iloc[-1]

    sub_indices = {
        "PM2.5": float(subindex_pm25(latest["PM2.5"])),
        "PM10": float(subindex_pm10(latest["PM10"])),
        "NO2": float(subindex_no2(latest["NO2"])),
        "Ozone": float(subindex_o3(latest["Ozone"])),
        "CO": float(subindex_co(latest["CO"]))
    }

    dominant = str(max(sub_indices, key=sub_indices.get))

    df["baseline_aqi"] = df.apply(
        lambda r: max(
            subindex_pm25(r["PM2.5"]),
            subindex_pm10(r["PM10"]),
            subindex_no2(r["NO2"]),
            subindex_o3(r["Ozone"]),
            subindex_co(r["CO"])
        ),
        axis=1
    )

    for col in TRAINED_CITY_COLUMNS:
        df[col] = 0

    df = df[scaler.feature_names_in_]

    predicted_series = []
    current_window = df.copy()

    for step in range(FORECAST_HOURS):

        X_step = scaler.transform(current_window)
        X_step = X_step.reshape(1, SEQ_LEN, -1)

        res_lstm = float(predict_lstm(X_step)[0][0])
        res_bilstm = float(predict_bilstm(X_step)[0][0])
        res_gru = float(predict_gru(X_step)[0][0])

        residual = (
            ENSEMBLE_WEIGHTS["LSTM"] * res_lstm +
            ENSEMBLE_WEIGHTS["BiLSTM"] * res_bilstm +
            ENSEMBLE_WEIGHTS["GRU"] * res_gru
        )

        hour = (now.hour + step) % 24

        if 8 <= hour <= 20:
            residual += 2
        else:
            residual -= 1

        next_aqi = float(current_window["baseline_aqi"].iloc[-1]) + (residual * 0.7)
        next_aqi = max(20, min(500, next_aqi))

        predicted_series.append(float(next_aqi))

        new_row = current_window.iloc[-1].astype(float).copy()
        new_row["baseline_aqi"] = float(next_aqi)

        current_window = pd.concat(
            [current_window.iloc[1:], pd.DataFrame([new_row])],
            ignore_index=True
        )

    predicted_series = np.array(predicted_series)

    forecast_24h = predicted_series[:24]
    forecast_48h = predicted_series[24:48]
    forecast_72h = predicted_series[48:72]

    fut_start = now.date()
    fut_end = (now + timedelta(hours=72)).date()

    wx_forecast = fetch_weather(fut_start, fut_end, lat, lon, forecast=True).iloc[:72]

    reasoning = analyze_priority_and_dominance(
        dominant,
        sub_indices,
        forecast_24h,
        wx_forecast
    )

    update_24hr_knowledge_graph(
        current_sub_indices=sub_indices,
        dominant_pollutant=dominant,
        reasoning_text=str(reasoning),
        predicted_aqi_24h=forecast_24h.tolist()
    )

    explanation = generate_explanation_report()

    try:
        graph_data = extract_graph_for_dashboard()
    except Exception as e:
        print("Graph extraction failed:", e)
        graph_data = {"nodes": [], "edges": []}

    result = {
        "location": {"latitude": float(lat), "longitude": float(lon)},
        "dominant_pollutant": dominant,
        "priority_analysis": reasoning,
        "predicted_aqi_next_24h": float(forecast_24h[-1]),
        "forecast_24h": forecast_24h.tolist(),
        "forecast_48h": forecast_48h.tolist(),
        "forecast_72h": forecast_72h.tolist(),
        "forecast_full": predicted_series.tolist(),
        "sub_indices": {k: float(v) for k, v in sub_indices.items()},
        "explanation_report": str(explanation),
        "rdf_graph": graph_data
    }
    _cache[key] = {
        "data": result,
        "ts": now_ts
    }

    print("💾 CACHE STORED")
    return result