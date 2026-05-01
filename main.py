from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aqi_pipeline import run_aqi_pipeline

app = FastAPI(title="AQI Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "AQI FastAPI running"}

@app.get("/aqi")
def get_aqi(lat: float, lon: float):

    result = run_aqi_pipeline(lat, lon)

    return result
