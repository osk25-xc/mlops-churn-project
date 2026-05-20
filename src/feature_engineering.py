# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

os.makedirs("models", exist_ok=True)


def load_and_clean(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop("customerID", axis=1, inplace=True)
    return df


def encode_features(df):
    df = df.copy()

    # Binary label encoding
    binary_cols = [
        "gender", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # One-hot encode multi-category columns
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    return df


def add_business_features(df):
    df = df.copy()
    # Revenue-per-month ratio
    df["charge_ratio"] = df["TotalCharges"] / (df["tenure"] + 1)
    # High value flag
    df["high_value"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    # Long tenure flag
    df["long_tenure"] = (df["tenure"] > 24).astype(int)
    return df


def scale_features(df):
    df = df.copy()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "charge_ratio"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved: models/scaler.pkl")
    return df, scaler


def get_train_test(df, target="Churn", test_size=0.2, random_state=42):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Save feature names for API use
    joblib.dump(X_train.columns.tolist(), "models/feature_names.pkl")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run_pipeline(path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    df = load_and_clean(path)
    df = encode_features(df)
    df = add_business_features(df)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = get_train_test(df)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = run_pipeline()
    print("\nFeature engineering complete.")
    print(f"Features: {X_train.shape[1]}")
