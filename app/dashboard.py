# app/dashboard.py
import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
)

# ─── Sidebar — Customer Input ─────────────────────────────────────────────────
st.sidebar.title("📋 Customer Details")
st.sidebar.markdown("---")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
total = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly * tenure), step=10.0)

st.sidebar.markdown("### Service & Contract")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
)
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

predict_btn = st.sidebar.button("🔮 Predict Churn", use_container_width=True)

# ─── Main Area ────────────────────────────────────────────────────────────────
st.title("🔁 Customer Churn Prediction Dashboard")
st.markdown("Enter customer details in the sidebar and click **Predict Churn**.")

if predict_btn:
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,
        "PhoneService": 1 if phone_service == "Yes" else 0,
        "PaperlessBilling": 1 if paperless == "Yes" else 0,
        "Contract_One_year": 1 if contract == "One year" else 0,
        "Contract_Two_year": 1 if contract == "Two year" else 0,
        "InternetService_Fiber_optic": 1 if internet == "Fiber optic" else 0,
        "InternetService_No": 1 if internet == "No" else 0,
        "PaymentMethod_Credit_card": 1 if "Credit card" in payment else 0,
        "PaymentMethod_Electronic_check": 1 if "Electronic check" in payment else 0,
        "PaymentMethod_Mailed_check": 1 if "Mailed check" in payment else 0,
        "MultipleLines_No_phone_service": 0,
        "MultipleLines_Yes": 0,
        "OnlineSecurity_No_internet_service": 1 if internet == "No" else 0,
        "OnlineSecurity_Yes": 0,
        "OnlineBackup_No_internet_service": 1 if internet == "No" else 0,
        "OnlineBackup_Yes": 0,
        "DeviceProtection_No_internet_service": 1 if internet == "No" else 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No_internet_service": 1 if internet == "No" else 0,
        "TechSupport_Yes": 0,
        "StreamingTV_No_internet_service": 1 if internet == "No" else 0,
        "StreamingTV_Yes": 0,
        "StreamingMovies_No_internet_service": 1 if internet == "No" else 0,
        "StreamingMovies_Yes": 0,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        res = response.json()

        prob = res["churn_probability"]
        prediction = res["churn_prediction"]
        risk = res["risk_level"]

        # ─── Result Cards ───
        col1, col2, col3 = st.columns(3)

        with col1:
            label = "⚠️ Will Churn" if prediction == 1 else "✅ Will Stay"
            color = "red" if prediction == 1 else "green"
            st.metric("Prediction", label)

        with col2:
            st.metric("Churn Probability", f"{prob * 100:.1f}%")

        with col3:
            risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(risk, "")
            st.metric("Risk Level", f"{risk_emoji} {risk}")

        st.markdown("---")

        # ─── Gauge Chart ───
        col_gauge, col_bar = st.columns(2)

        with col_gauge:
            gauge_color = "#e53935" if prob > 0.5 else "#1e88e5"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                delta={"reference": 50},
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 40], "color": "#c8e6c9"},
                        {"range": [40, 70], "color": "#fff9c4"},
                        {"range": [70, 100], "color": "#ffcdd2"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            ))
            fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_bar:
            # Key customer feature summary
            features = {
                "Tenure": tenure,
                "Monthly ($)": monthly,
                "Total ($)": total / 100,
            }
            fig_bar = px.bar(
                x=list(features.keys()),
                y=list(features.values()),
                title="Customer Profile",
                color_discrete_sequence=["steelblue"],
            )
            fig_bar.update_layout(height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)

        # ─── Advice ───
        st.markdown("### 💡 Recommended Actions")
        if risk == "High":
            st.error("This customer is at HIGH risk of churning. Consider offering a discount or contract upgrade immediately.")
        elif risk == "Medium":
            st.warning("This customer is at MEDIUM risk. Consider a loyalty reward or proactive outreach.")
        else:
            st.success("This customer is at LOW risk. Keep up the good service!")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure FastAPI is running on http://localhost:8000")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ─── Prediction Logs ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Recent Predictions Log")

log_path = "logs/prediction_log.csv"
if os.path.exists(log_path):
    try:
        cols = list(payload.keys()) + ["churn_prediction", "churn_probability", "risk_level", "timestamp"]
        df_log = pd.read_csv(log_path, names=cols)
        display_cols = ["timestamp", "tenure", "MonthlyCharges", "churn_prediction", "churn_probability", "risk_level"]
        st.dataframe(df_log[display_cols].tail(20), use_container_width=True)

        churn_counts = df_log["risk_level"].value_counts().reset_index()
        churn_counts.columns = ["Risk Level", "Count"]
        fig_pie = px.pie(churn_counts, names="Risk Level", values="Count",
                         color="Risk Level",
                         color_discrete_map={"High": "#e53935", "Medium": "#fdd835", "Low": "#43a047"},
                         title="Risk Distribution (All Predictions)")
        st.plotly_chart(fig_pie, use_container_width=True)
    except Exception:
        st.info("Log file exists but could not be parsed yet.")
else:
    st.info("No predictions logged yet. Make a prediction above to get started.")
