# src/train.py
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from feature_engineering import run_pipeline
import os

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def save_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    path = f"logs/cm_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def train_all(data_path="data/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    X_train, X_test, y_train, y_test, scaler = run_pipeline(data_path)
    models = get_models()

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("churn-prediction")

    best_model = None
    best_model_name = ""
    best_auc = 0.0
    all_results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            # Log to MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_metric("accuracy", metrics["accuracy"])
            mlflow.log_metric("f1_score", metrics["f1_score"])
            mlflow.log_metric("auc_roc", metrics["auc_roc"])

            # Save confusion matrix as artifact
            cm_path = save_confusion_matrix(y_test, metrics["y_pred"], name)
            mlflow.log_artifact(cm_path)

            # Log model
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            all_results[name] = {
                "accuracy": round(metrics["accuracy"], 4),
                "f1_score": round(metrics["f1_score"], 4),
                "auc_roc": round(metrics["auc_roc"], 4),
            }

            print(f"  Accuracy : {metrics['accuracy']:.4f}")
            print(f"  F1 Score : {metrics['f1_score']:.4f}")
            print(f"  AUC-ROC  : {metrics['auc_roc']:.4f}")
            print(classification_report(y_test, metrics["y_pred"],
                                        target_names=["Stay", "Churn"]))

            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_model = model
                best_model_name = name

    # Save best model
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\nBest model: {best_model_name} (AUC={best_auc:.4f})")
    print("Saved: models/best_model.pkl")

    # Summary table
    results_df = pd.DataFrame(all_results).T
    print(f"\n{'='*45}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*45}")
    print(results_df.to_string())

    return best_model, all_results


if __name__ == "__main__":
    train_all()
