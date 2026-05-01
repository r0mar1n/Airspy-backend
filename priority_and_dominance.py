import numpy as np

# =========================================================
# Meteorological primitives
# =========================================================

def is_stagnant(avg_wind, total_rain):
    return avg_wind < 3 and total_rain == 0

def is_dispersion_favorable(avg_wind, total_rain):
    return avg_wind >= 5 or total_rain > 0

def is_photochemical(max_temp):
    return max_temp >= 30

# =========================================================
# Core reasoning engine
# =========================================================

def analyze_priority_and_dominance(
    current_dominant_pollutant: str,
    sub_indices: dict,
    predicted_aqi_series: np.ndarray,
    forecast_weather_df
):
    """
    Inputs:
    - current_dominant_pollutant : str (at T0)
    - sub_indices                : dict (at T0)
    - predicted_aqi_series       : np.ndarray (length 24)
    - forecast_weather_df        : DataFrame (24 rows, future)

    Output:
    - reasoning dict (JSON-serializable)
    """

    # -------------------------------
    # AQI trajectory analysis
    # -------------------------------
    aqi_trend_value = predicted_aqi_series[-1] - predicted_aqi_series[0]
    rising = aqi_trend_value > 15

    # -------------------------------
    # Forecast weather summary
    # -------------------------------
    avg_wind = forecast_weather_df["windspeed_10m"].mean()
    total_rain = forecast_weather_df["rain"].sum()
    max_temp = forecast_weather_df["temperature_2m"].max()

    # -------------------------------
    # Priority reasoning rules
    # -------------------------------
    priority = current_dominant_pollutant
    reason = "DOMINANT_PERSISTENCE"

    if current_dominant_pollutant in ["PM2.5", "PM10"]:
        if is_stagnant(avg_wind, total_rain) and rising:
            priority = "PM2.5"
            reason = "STAGNATION_PM_ACCUMULATION"
        elif is_dispersion_favorable(avg_wind, total_rain):
            priority = "Ozone"
            reason = "PM_DISPERSION_OZONE_RISK"

    if current_dominant_pollutant == "NO2" and is_photochemical(max_temp):
        priority = "Ozone"
        reason = "PHOTOCHEMICAL_OZONE_FORMATION"

    if max_temp < 20 and avg_wind < 3:
        priority = "PM2.5"
        reason = "WINTER_PM_PERSISTENCE"

    # -------------------------------
    # Risk level (based on future AQI)
    # -------------------------------
    max_aqi = predicted_aqi_series.max()

    if max_aqi <= 50:
        risk = "Good"
    elif max_aqi <= 100:
        risk = "Satisfactory"
    elif max_aqi <= 200:
        risk = "Moderate"
    elif max_aqi <= 300:
        risk = "Poor"
    elif max_aqi <= 400:
        risk = "Very Poor"
    else:
        risk = "Severe"

    return {
        "dominant_pollutant": current_dominant_pollutant,
        "priority_pollutant": priority,
        "reason_code": reason,
        "risk_window": "Next_24_Hours",
        "risk_level": risk,
        "aqi_trend": "Rising" if rising else "Stable/Declining"
    }
