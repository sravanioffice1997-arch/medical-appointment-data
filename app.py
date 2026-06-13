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

forecast_model = joblib.load("02_demand_forecast_rf.pkl")
no_show_model = joblib.load("03_no_show_xgb_model.pkl")

classification_features = joblib.load(
    "05_classification_features.pkl"
)

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

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=40
        )

        gender = st.selectbox(
            "Gender",
            ["F", "I", "M"]
        )

        appointment_shift = st.selectbox(
            "Appointment Shift",
            ["morning", "afternoon"]
        )

        diabetes = st.selectbox(
            "Diabetes",
            [0,1]
        )

        hipertension = st.selectbox(
            "Hypertension",
            [0,1]
        )

        alcoholism = st.selectbox(
            "Alcoholism",
            [0,1]
        )

    with col2:

        handcap = st.selectbox(
            "Handicap",
            [0,1]
        )

        scholarship = st.selectbox(
            "Scholarship",
            [0,1]
        )

        sms_received = st.selectbox(
            "SMS Received",
            [0,1]
        )

        rain_intensity = st.selectbox(
            "Rain Intensity",
            ["heavy","moderate","no_rain","weak"]
        )

        heat_intensity = st.selectbox(
            "Heat Intensity",
            ["cold","heavy_cold","heavy_warm","mild","warm"]
        )

    if st.button("Predict No Show"):

        input_df = pd.DataFrame({
            "specialty":[0],
            "appointment_time":[9],
            "gender":[{"F":0,"I":1,"M":2}[gender]],
            "disability":[0],
            "place":[0],
            "appointment_shift":[{"afternoon":0,"morning":1}[appointment_shift]],
            "age":[age],
            "under_12_years_old":[1 if age < 12 else 0],
            "over_60_years_old":[1 if age > 60 else 0],
            "patient_needs_companion":[0],
            "average_temp_day":[25],
            "average_rain_day":[0],
            "max_temp_day":[30],
            "max_rain_day":[0],
            "rainy_day_before":[0],
            "storm_day_before":[0],
            "rain_intensity":[
                {
                    "heavy":0,
                    "moderate":1,
                    "no_rain":2,
                    "weak":3
                }[rain_intensity]
            ],
            "heat_intensity":[
                {
                    "cold":0,
                    "heavy_cold":1,
                    "heavy_warm":2,
                    "mild":3,
                    "warm":4
                }[heat_intensity]
            ],
            "Hipertension":[hipertension],
            "Diabetes":[diabetes],
            "Alcoholism":[alcoholism],
            "Handcap":[handcap],
            "Scholarship":[scholarship],
            "SMS_received":[sms_received],
            "year":[2021],
            "month":[6],
            "day":[15],
            "weekday":[2],
            "week":[24]
        })

        input_df = input_df[classification_features]

        prediction = no_show_model.predict(input_df)[0]

        probability = no_show_model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(
                f"Likely No-Show (Probability = {probability:.2%})"
            )
        else:
            st.success(
                f"Likely Will Attend (Probability = {1-probability:.2%})"
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
