# src/monitoring.py
import pandas as pd
import numpy as np
from scipy import stats
import logging
import json
from datetime import datetime
import os

os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/monitoring.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "charge_ratio"]


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).
    PSI < 0.1  → no significant change
    PSI 0.1–0.2 → moderate change (monitor)
    PSI > 0.2  → significant change (retrain!)
    """
    def _bucketize(arr, bins):
        counts, _ = np.histogram(arr, bins=bins)
        counts = counts / len(arr)
        counts = np.where(counts == 0, 1e-4, counts)
        return counts

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    bins = np.linspace(min_val, max_val, buckets + 1)

    expected_pct = _bucketize(expected, bins)
    actual_pct = _bucketize(actual, bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return round(float(psi), 4)


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """
    Run KS-test and PSI on all numeric features.
    Returns a dict with drift flags per feature.
    """
    results = {}

    for feature in NUMERIC_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref = reference_df[feature].dropna().values
        cur = current_df[feature].dropna().values

        ks_stat, ks_p = stats.ks_2samp(ref, cur)
        psi = compute_psi(ref, cur)

        drift_flag = ks_p < 0.05 or psi > 0.2

        results[feature] = {
            "ks_statistic": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_p), 4),
            "psi": psi,
            "drift_detected": drift_flag,
        }

        status = "⚠️ DRIFT" if drift_flag else "✅ OK"
        print(f"  {feature}: KS p={ks_p:.4f} | PSI={psi:.4f} | {status}")

    logger.info(json.dumps({"timestamp": datetime.utcnow().isoformat(), "drift_report": results}))
    return results


def check_prediction_drift(log_csv: str = "logs/prediction_log.csv",
                             window: int = 200) -> dict:
    """
    Compare recent predictions against the oldest window to detect output drift.
    """
    try:
        df = pd.read_csv(log_csv)
        if len(df) < window * 2:
            print("Not enough predictions for drift analysis yet.")
            return {}

        reference = df.head(window)
        current = df.tail(window)

        ref_rate = reference["churn_prediction"].mean()
        cur_rate = current["churn_prediction"].mean()

        drift = abs(cur_rate - ref_rate) > 0.10

        result = {
            "reference_churn_rate": round(ref_rate, 4),
            "current_churn_rate": round(cur_rate, 4),
            "delta": round(cur_rate - ref_rate, 4),
            "output_drift_detected": drift,
        }

        print(f"\nOutput drift check:")
        print(f"  Reference churn rate : {ref_rate:.4f}")
        print(f"  Current churn rate   : {cur_rate:.4f}")
        print(f"  Delta                : {cur_rate - ref_rate:+.4f}")
        print(f"  Drift detected       : {'⚠️ YES' if drift else '✅ NO'}")

        logger.info(json.dumps({"timestamp": datetime.utcnow().isoformat(), "output_drift": result}))
        return result

    except FileNotFoundError:
        print("Prediction log not found.")
        return {}


def api_performance_summary(log_csv: str = "logs/prediction_log.csv") -> dict:
    """Print a quick summary of logged predictions."""
    try:
        df = pd.read_csv(log_csv)
        summary = {
            "total_predictions": len(df),
            "churn_rate": round(df["churn_prediction"].mean(), 4),
            "avg_probability": round(df["churn_probability"].mean(), 4),
            "high_risk_pct": round((df["risk_level"] == "High").mean(), 4),
        }
        print("\nAPI Performance Summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return summary
    except Exception as e:
        print(f"Could not compute summary: {e}")
        return {}


if __name__ == "__main__":
    print("Running monitoring checks...\n")
    api_performance_summary()
    check_prediction_drift()
