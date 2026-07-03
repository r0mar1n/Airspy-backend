# AirSpy: Explainable AQI Forecasting System (Backend)

🔗 **Live Demo:** https://aqi-prediction-system-with-rdf-expl.vercel.app/

🔗 **Frontend Repository:** https://github.com/r0mar1n/AirSpy

AirSpy is an end-to-end air quality forecasting system that predicts AQI trends across Indian cities up to **72 hours ahead** — and explains *why*, not just *what*. A deep learning ensemble generates the forecast, while an RDF knowledge graph layer traces each prediction back to the pollutants and environmental conditions driving it.

> This repository contains the **backend** of the system. It powers the live application by handling data preprocessing, AQI prediction, RDF knowledge graph generation, and REST APIs consumed by the React frontend. The frontend is maintained in a separate public repository linked above.

---

# How the system works

1. **The frontend sends a prediction request** to the FastAPI backend.
2. **Historical weather and pollutant data are preprocessed** before being passed into a weighted deep learning ensemble consisting of **LSTM, BiLSTM, and GRU** models.
3. **The ensemble forecasts AQI** for the next 72 hours using models trained on **26,000+ CPCB records** collected across four Indian cities throughout 2024, allowing the model to capture seasonal and geographical variations.
4. **An RDF knowledge graph** is generated around the prediction, linking the forecasted AQI to the pollutants and environmental conditions contributing to it.
5. **The backend returns** the AQI forecast, pollutant breakdown, and explainability data through REST APIs, which are visualized by the frontend dashboard.

### Model Performance

- **85–90% AQI Category Accuracy**
- **80–88% Trend Direction Accuracy**

Like any real-world forecasting model, predictions may occasionally differ from actual conditions, particularly for locations or environmental scenarios that are underrepresented in the training data.

---

# What this repository contains

This repository contains the complete backend powering AirSpy, including:

- FastAPI REST APIs
- Deep learning inference pipeline
- AQI preprocessing pipeline
- Weighted LSTM–BiLSTM–GRU ensemble model
- RDF knowledge graph generation
- SPARQL query processing
- Explainability layer
- Backend deployment configuration for Render

---

# Tech Stack

- Python
- FastAPI
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- RDFLib
- SPARQL
- Uvicorn

---

# Project Structure

AirSpy is split into two repositories.

## Frontend

React + TypeScript dashboard responsible for:

- User interface
- AQI visualization
- RDF graph visualization
- API integration

**Repository:** https://github.com/r0mar1n/AirSpy

**Deployment:** Vercel

---

## Backend (This Repository)

Responsible for:

- Data preprocessing
- Deep learning inference
- AQI forecasting
- RDF knowledge graph generation
- Explainability APIs
- REST API endpoints

**Deployment:** Render

---

# Architecture

![AirSpy Architecture](./new_flochart.jpeg)

---

# Running Locally

## Requirements

- Python 3.9+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/r0mar1n/AirSpy-Backend.git
cd AirSpy-Backend
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the FastAPI server

```bash
uvicorn main:app --reload
```

### 6. Open the interactive API documentation

```
http://127.0.0.1:8000/docs
```

---

# Project Requirements

- Keep the **models/** folder in the project root.
- Ensure all trained model files are present before starting the server.
- Python 3.9 or newer is recommended.

---

# API

## Main Endpoint

```
POST /predict
```

Returns:

- Predicted AQI
- 72-hour AQI forecast
- Pollutant contribution analysis
- RDF explainability data

Additional API endpoints are available through the automatically generated FastAPI documentation.

---

# Live Demo

The complete AirSpy application is publicly deployed.

🔗 **https://aqi-prediction-system-with-rdf-expl.vercel.app/**

### Deployment Architecture

**Frontend**
- React + TypeScript
- Hosted on **Vercel**

**Backend**
- FastAPI
- Hosted on **Render**

The frontend communicates with this backend through REST APIs to generate live AQI forecasts and explainable insights.

---

# Related Repository

The frontend dashboard for AirSpy is available here:

🔗 **https://github.com/r0mar1n/AirSpy**
