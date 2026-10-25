# Customer Churn Prediction | MLOps Pipeline

An end-to-end machine-learning engineering project structured around churn prediction, experiment tracking, an API, a lightweight user interface, and operational monitoring.

## Objective
Demonstrate a reproducible workflow for training and serving a churn model, with supporting components for experiment tracking and monitoring.

## Architecture & Components
- **Exploratory analysis:** `src/eda.py`
- **Feature engineering:** `src/feature_engineering.py`
- **Model training:** `src/train.py`
- **Experiment tracking:** MLflow
- **Prediction API:** FastAPI
- **Interactive interface:** Streamlit
- **Containerization:** Docker / Docker Compose
- **Monitoring:** prediction logs and drift-related checks
- **Tests:** `tests/`

## Technology Stack
Python · XGBoost · FastAPI · Streamlit · MLflow · Docker

## Setup
Create and activate a virtual environment, then install the repository dependencies:

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The project instructions reference the Telco Customer Churn dataset. Place the dataset at the path expected by the scripts before running the pipeline.

## Run Components
```bash
python src/eda.py
python src/train.py
mlflow ui
uvicorn app.main:app --reload
streamlit run app/dashboard.py
```

Use the repository's Docker Compose configuration if you prefer a containerized setup. Review the code and configuration before deploying anywhere.

## Important Notes
This is a portfolio/demo project, not a claim of production deployment or validated business performance. Model quality, security, data handling, and monitoring behavior should be evaluated in the target environment before real-world use.

## Skills Demonstrated
ML pipeline structure · Feature engineering · Model training · Experiment tracking · API development · Containerization · Monitoring concepts

---
Created by **Omkar S. Kshirsagar** | [GitHub Profile](https://github.com/osk25-xc)
