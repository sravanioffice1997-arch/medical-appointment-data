import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Medical Appointment Analytics",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================

forecast_model = joblib.load("demand_forecast_rf.pkl")

# =========================
# SIDEBAR
# =========================

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "No-Show Prediction",
        "Demand Forecasting"
    ]
)

# =========================
# HOME PAGE
# =========================

if menu == "Home":

    st.title("🏥 Medical Appointment Analytics")

    st.markdown("""
    ### Project Overview

    This dashboard was developed to support healthcare appointment management.

    ### Machine Learning Models

    **No-Show Prediction**
    - XGBoost Classifier
    - F1 Score: 0.629
    - ROC-AUC: 0.783

    **Demand Forecasting**
    - Random Forest Regressor
    - MAE: 135.46
    - RMSE: 230.57
    - R²: 0.34
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="No-Show ROC-AUC",
            value="0.783"
        )

    with col2:
        st.metric(
            label="Forecasting R²",
            value="0.340"
        )
# =========================
# NO-SHOW PREDICTION PAGE
# =========================

elif menu == "No-Show Prediction":

    st.title("🤖 No-Show Prediction")

    st.write(
        "Predict whether a patient is likely to miss an appointment."
    )

    st.info(
        "No-Show Prediction module will be connected to the XGBoost model."
    )
# =========================
# FORECASTING PAGE
# =========================

elif menu == "Demand Forecasting":

    st.title("📈 Appointment Demand Forecasting")

    st.write(
        "Enter historical appointment information to estimate future demand."
    )

    col1, col2 = st.columns(2)

    with col1:

        dayofweek = st.selectbox(
            "Day of Week",
            [0,1,2,3,4,5,6]
        )

        month = st.slider(
            "Month",
            1,
            12,
            1
        )

        quarter = st.selectbox(
            "Quarter",
            [1,2,3,4]
        )

        day = st.slider(
            "Day",
            1,
            31,
            1
        )

    with col2:

        lag_1 = st.number_input(
            "Appointments Yesterday",
            min_value=0,
            value=200
        )

        lag_7 = st.number_input(
            "Appointments 7 Days Ago",
            min_value=0,
            value=180
        )

        lag_14 = st.number_input(
            "Appointments 14 Days Ago",
            min_value=0,
            value=190
        )

        rolling_mean_7 = st.number_input(
            "7-Day Rolling Average",
            min_value=0.0,
            value=200.0
        )

        rolling_mean_14 = st.number_input(
            "14-Day Rolling Average",
            min_value=0.0,
            value=210.0
        )

    if st.button("Forecast Demand"):

        input_df = pd.DataFrame({
            "dayofweek":[dayofweek],
            "month":[month],
            "quarter":[quarter],
            "day":[day],
            "lag_1":[lag_1],
            "lag_7":[lag_7],
            "lag_14":[lag_14],
            "rolling_mean_7":[rolling_mean_7],
            "rolling_mean_14":[rolling_mean_14]
        })

        prediction = forecast_model.predict(input_df)

        st.success(
    f"Predicted Appointments: {int(prediction[0])}"
)
