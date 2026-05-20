# MLOps Customer Churn Prediction Pipeline

A production-ready end-to-end MLOps project using XGBoost, FastAPI, MLflow, Docker, and Streamlit.

## Project Structure

```
mlops-churn-project/
├── app/
│   ├── main.py          # FastAPI prediction API
│   └── dashboard.py     # Streamlit UI
├── src/
│   ├── eda.py           # Exploratory data analysis
│   ├── feature_engineering.py
│   ├── train.py         # Model training + MLflow tracking
│   └── monitoring.py    # Drift detection & logging
├── models/              # Saved model artifacts (.pkl)
├── logs/                # Prediction logs & plots
├── mlruns/              # MLflow experiment tracking
├── data/                # Raw dataset (not committed)
├── tests/
├── Dockerfile
├── docker-compose.yml
├── render.yaml
└── requirements.txt
```

## Quick Start

### 1. Setup environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download dataset
Download **Telco Customer Churn** from Kaggle and place it at:
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

### 3. Run EDA
```bash
python src/eda.py
```

### 4. Train models (logs to MLflow)
```bash
python src/train.py
```

### 5. View MLflow dashboard
```bash
mlflow ui
# Open http://localhost:5000
```

### 6. Start FastAPI
```bash
uvicorn app.main:app --reload
# API docs at http://localhost:8000/docs
```

### 7. Start Streamlit dashboard
```bash
streamlit run app/dashboard.py
# Open http://localhost:8501
```

### 8. Run with Docker
```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root |
| GET | `/health` | Health check |
| POST | `/predict` | Predict churn |
| GET | `/logs/summary` | Prediction summary |

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9 |
| ML | XGBoost, Scikit-learn |
| Tracking | MLflow |
| API | FastAPI |
| Frontend | Streamlit |
| Container | Docker |
| Deployment | Render |

## Resume Line
> Developed a production-ready customer churn prediction system using XGBoost, FastAPI, MLflow, Docker, and Streamlit with end-to-end deployment and monitoring.
