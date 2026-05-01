import pandas as pd
import numpy as np
from pathlib import Path

# ------------------ PATHS ------------------
RAW_PATH = Path("Combined_AQ_Weather_NO2_AQI_2024_Robust.csv")
OUT_PATH = Path("data/processed/AQI_Model_Features_v2.csv")

# ------------------ LOAD DATA ------------------
df = pd.read_csv(RAW_PATH)
print("Original dataset shape:", df.shape)

# ------------------ COLUMNS TO KEEP ------------------
COLUMNS = [
    "Datetime", "City",
    "PM2.5", "PM10", "CO", "Ozone", "NO2",
    "temperature_2m", "relative_humidity_2m",
    "precipitation", "rain", "pressure_msl",
    "windspeed_10m", "winddirection_10m",
    "Final_AQI"
]

df = df[COLUMNS]

# ------------------ DATETIME ------------------
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(["City", "Datetime"]).reset_index(drop=True)

# ------------------ BASIC CLEANING ------------------
pollutants = ["PM2.5", "PM10", "CO", "Ozone", "NO2"]
for col in pollutants:
    df[col] = df[col].clip(lower=0)

df = (
    df
    .groupby("City", group_keys=False)
    .apply(lambda g: g.ffill())
)

# ------------------ SEASONAL CONTEXT ------------------
df["month"] = df["Datetime"].dt.month
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df.drop(columns=["month"], inplace=True)

# ------------------ CPCB SUB-INDEX FUNCTIONS ------------------
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

def subindex_co(c):
    if c <= 1: return linear_subindex(c, 0, 1, 0, 50)
    if c <= 2: return linear_subindex(c, 1.1, 2, 51, 100)
    if c <= 10: return linear_subindex(c, 2.1, 10, 101, 200)
    if c <= 17: return linear_subindex(c, 10.1, 17, 201, 300)
    if c <= 34: return linear_subindex(c, 17.1, 34, 301, 400)
    return linear_subindex(c, 34.1, 50, 401, 500)

def subindex_o3(c):
    if c <= 50: return linear_subindex(c, 0, 50, 0, 50)
    if c <= 100: return linear_subindex(c, 51, 100, 51, 100)
    if c <= 168: return linear_subindex(c, 101, 168, 101, 200)
    if c <= 208: return linear_subindex(c, 169, 208, 201, 300)
    if c <= 748: return linear_subindex(c, 209, 748, 301, 400)
    return linear_subindex(c, 749, 1000, 401, 500)

# ------------------ COMPUTE BASELINE AQI ------------------
df["sub_pm25"] = df["PM2.5"].apply(subindex_pm25)
df["sub_pm10"] = df["PM10"].apply(subindex_pm10)
df["sub_no2"] = df["NO2"].apply(subindex_no2)
df["sub_co"] = df["CO"].apply(subindex_co)
df["sub_o3"] = df["Ozone"].apply(subindex_o3)

df["baseline_aqi"] = df[
    ["sub_pm25", "sub_pm10", "sub_no2", "sub_co", "sub_o3"]
].max(axis=1)

# ------------------ RESIDUAL TARGET ------------------
df["aqi_residual"] = df["Final_AQI"] - df["baseline_aqi"]

# ------------------ CLEANUP ------------------
df.drop(columns=[
    "sub_pm25", "sub_pm10", "sub_no2", "sub_co", "sub_o3"
], inplace=True)

# ------------------ FINAL CHECK ------------------
assert df.isnull().sum().sum() == 0, "⚠️ Missing values remain!"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PATH, index=False)

print("Processed dataset shape:", df.shape)
print(f"✅ Saved physics-informed dataset to: {OUT_PATH}")
