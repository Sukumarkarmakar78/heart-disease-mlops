import joblib
import logging
import sys
import time

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import (
    Counter,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
)
from pythonjsonlogger import jsonlogger
from starlette.responses import Response

try:
    try:
        import numpy._core as numpy_core  # type: ignore
    except ModuleNotFoundError:
        import numpy.core as numpy_core  # type: ignore
    sys.modules.setdefault("numpy._core", numpy_core)
except Exception:
    pass

# ============================
# LOAD MODEL PIPELINE
# ============================
model = joblib.load("models/model_pipeline.pkl")

# ----------------------------
# Logging configuration
# ----------------------------
logger = logging.getLogger("heart_api")
logger.setLevel(logging.INFO)
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s'
)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

# ----------------------------
# Prometheus metrics
# ----------------------------
REQUEST_COUNT = Counter(
    "request_count",
    "Total HTTP requests",
    ["method", "endpoint", "http_status"],
)
REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Request latency",
    ["method", "endpoint"],
)

app = FastAPI()


# ============================
# INPUT SCHEMA
# ============================

class HeartData(BaseModel):
    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ============================
# TEST ENDPOINT
# ============================

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API Running"}


# Middleware for logging and metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    path = request.url.path
    method = request.method
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        logger.exception("Unhandled exception during request")
        raise
    finally:
        elapsed = time.time() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)
        REQUEST_COUNT.labels(
            method=method,
            endpoint=path,
            http_status=str(status_code),
        ).inc()
        logger.info(
            "request_handled",
            extra={
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration": elapsed,
            },
        )
    return response


@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================
# HEALTH CHECK
# ============================
@app.get("/health")
def health():
    return {"status": "OK"}


# ============================
# PREDICTION ENDPOINT
# ============================
@app.post("/predict")
def predict(data: HeartData):

    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict using pipeline (no manual scaling required)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "confidence": float(probability),
    }
