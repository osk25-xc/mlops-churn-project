# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import logging
import json
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/api.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a customer will churn using XGBoost.",
    version="1.0.0",
)

# Load model artifacts on startup
model = None
scaler = None
feature_names = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_names
    try:
        model = joblib.load("models/best_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        logger.info("Model artifacts loaded successfully.")
        print("✅ Model artifacts loaded.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        print(f"❌ Error loading artifacts: {e}")


# ─── Request / Response Schemas ───────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    tenure: float = Field(..., ge=0, le=72, example=24)
    MonthlyCharges: float = Field(..., ge=0, example=75.5)
    TotalCharges: float = Field(..., ge=0, example=1812.0)
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male", example=1)
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0)
    Partner: int = Field(..., ge=0, le=1, description="0=No, 1=Yes", example=1)
    Dependents: int = Field(..., ge=0, le=1, example=0)
    PhoneService: int = Field(..., ge=0, le=1, example=1)
    PaperlessBilling: int = Field(..., ge=0, le=1, example=1)
    # Contract type (one-hot, drop_first=True baseline: Month-to-month)
    Contract_One_year: int = Field(0, ge=0, le=1, example=0)
    Contract_Two_year: int = Field(0, ge=0, le=1, example=0)
    # Internet service (baseline: DSL)
    InternetService_Fiber_optic: int = Field(0, ge=0, le=1, example=1)
    InternetService_No: int = Field(0, ge=0, le=1, example=0)
    # Payment method (baseline: Bank transfer)
    PaymentMethod_Credit_card: int = Field(0, ge=0, le=1, example=0)
    PaymentMethod_Electronic_check: int = Field(0, ge=0, le=1, example=1)
    PaymentMethod_Mailed_check: int = Field(0, ge=0, le=1, example=0)
    # Other binary service flags
    MultipleLines_No_phone_service: int = Field(0, ge=0, le=1, example=0)
    MultipleLines_Yes: int = Field(0, ge=0, le=1, example=0)
    OnlineSecurity_No_internet_service: int = Field(0, ge=0, le=1, example=0)
    OnlineSecurity_Yes: int = Field(0, ge=0, le=1, example=0)
    OnlineBackup_No_internet_service: int = Field(0, ge=0, le=1, example=0)
    OnlineBackup_Yes: int = Field(0, ge=0, le=1, example=0)
    DeviceProtection_No_internet_service: int = Field(0, ge=0, le=1, example=0)
    DeviceProtection_Yes: int = Field(0, ge=0, le=1, example=0)
    TechSupport_No_internet_service: int = Field(0, ge=0, le=1, example=0)
    TechSupport_Yes: int = Field(0, ge=0, le=1, example=0)
    StreamingTV_No_internet_service: int = Field(0, ge=0, le=1, example=0)
    StreamingTV_Yes: int = Field(0, ge=0, le=1, example=0)
    StreamingMovies_No_internet_service: int = Field(0, ge=0, le=1, example=0)
    StreamingMovies_Yes: int = Field(0, ge=0, le=1, example=0)


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    timestamp: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {"message": "Churn Prediction API is running. Visit /docs for Swagger UI."}


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    try:
        input_dict = customer.dict()

        # Compute engineered features
        charge_ratio = input_dict["TotalCharges"] / (input_dict["tenure"] + 1)
        high_value = int(input_dict["MonthlyCharges"] > 64.76)   # approx. median
        long_tenure = int(input_dict["tenure"] > 24)

        input_dict["charge_ratio"] = charge_ratio
        input_dict["high_value"] = high_value
        input_dict["long_tenure"] = long_tenure

        # Scale numeric columns
        num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charge_ratio"]
        df_input = pd.DataFrame([input_dict])

        df_input[num_cols] = scaler.transform(df_input[num_cols])

        # Align with training feature order
        if feature_names:
            for col in feature_names:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[feature_names]

        prediction = int(model.predict(df_input)[0])
        probability = float(model.predict_proba(df_input)[0][1])

        risk_level = (
            "High" if probability > 0.70
            else "Medium" if probability > 0.40
            else "Low"
        )

        ts = datetime.utcnow().isoformat()

        # Log prediction
        log_record = {
            **customer.dict(),
            "churn_prediction": prediction,
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "timestamp": ts,
        }
        logger.info(json.dumps(log_record))

        # Append to CSV log
        pd.DataFrame([log_record]).to_csv(
            "logs/prediction_log.csv", mode="a", header=False, index=False
        )

        return PredictionResponse(
            churn_prediction=prediction,
            churn_probability=round(probability, 4),
            risk_level=risk_level,
            timestamp=ts,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/summary", tags=["Monitoring"])
def log_summary():
    try:
        df = pd.read_csv("logs/prediction_log.csv")
        return {
            "total_predictions": len(df),
            "churn_rate": round(df["churn_prediction"].mean(), 4),
            "avg_probability": round(df["churn_probability"].mean(), 4),
            "risk_distribution": df["risk_level"].value_counts().to_dict(),
        }
    except FileNotFoundError:
        return {"message": "No predictions logged yet."}
