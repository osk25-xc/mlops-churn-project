# Customer Churn Prediction | MLOps Portfolio Project

An end-to-end portfolio project exploring how a customer-churn model can be developed, served, and monitored. The repository is organized around a reproducible machine-learning workflow and supporting application components.

## Project workflow

1. Explore the churn dataset and inspect its quality.
2. Prepare features for model training.
3. Train and evaluate a classification model.
4. Track experiments with MLflow.
5. Expose predictions through a FastAPI application.
6. Provide an interactive interface with Streamlit.
7. Package components with Docker and review monitoring-related logs.

## Technology

Python · Pandas · Scikit-learn / XGBoost (as implemented) · MLflow · FastAPI · Streamlit · Docker

## Repository components

- `src/eda.py` — exploratory analysis
- `src/feature_engineering.py` — feature preparation
- `src/train.py` — model training
- `app/` — API and interface components
- `tests/` — automated tests

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Use the dataset path expected by the project scripts. Review configuration and required environment variables before running services.

## Run

Run only the components available in the current checkout:

```bash
python src/eda.py
python src/train.py
mlflow ui
uvicorn app.main:app --reload
streamlit run app/dashboard.py
```

If a command or path differs in your local version, follow the actual project structure and configuration.

## Evaluation and limitations

Report model metrics only after running the training and evaluation workflow. This is a portfolio demonstration, not a claim of production deployment. Validate data permissions, model quality, security, and monitoring before any real-world use.

## Skills demonstrated

Machine-learning workflow · Feature engineering · Experiment tracking · API development · Containerization · Model monitoring concepts

Maintained by [Omkar S. Kshirsagar](https://github.com/osk25-xc).
