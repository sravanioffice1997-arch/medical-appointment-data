"""
CER Rehabilitation Center — No-Show Risk & Demand Forecasting Dashboard
Run with:  streamlit run app.py
"""

import json
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Appointment No-Show & Demand Dashboard",
    page_icon="🏥",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background-color: #F5F8FA;
        border: 1px solid #E1E8ED;
        border-radius: 10px;
        padding: 1rem 1.2rem;
    }
    .risk-low { color: #1E8E3E; font-weight: 700; }
    .risk-medium { color: #E8A33D; font-weight: 700; }
    .risk-high { color: #D93025; font-weight: 700; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Artifact loading
# ------------------------------------------------------------------
REQUIRED_FILES = ["model_bundle.pkl", "data_bundle.pkl"]


@st.cache_resource(show_spinner=False)
def load_models():
    bundle = joblib.load("model_bundle.pkl")
    return (bundle["classifier"], bundle["label_encoders"], bundle["classifier_features"],
            bundle["forecaster"], bundle["forecaster_features"])


@st.cache_data(show_spinner=False)
def load_tables():
    bundle = joblib.load("data_bundle.pkl")
    daily = bundle["daily_history"]
    insights = bundle["insights"]
    return {
        "daily": daily, "specialty": insights["specialty"], "shift": insights["shift"],
        "rain": insights["rain"], "heat": insights["heat"], "gender": insights["gender"],
        "specialty_share": bundle["specialty_share"], "feature_importance": bundle["feature_importance"],
        "stats": bundle["overview_stats"],
    }


import os
missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
if missing:
    st.error(
        "Missing required file(s): " + ", ".join(missing) +

        "\n\nRun the notebook's **Export Artifacts for Streamlit App** section, "
        "download the files it produces, and place them in the same folder as `app.py`."
    )
    st.stop()

clf, le_dict, clf_features, reg, reg_features = load_models()
tables = load_tables()
daily = tables["daily"]
stats = tables["stats"]


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("🏥 Appointment No-Show & Demand Dashboard")
st.caption("Decision support for clinic staff — no-show risk scoring and short-term demand forecasting.")

tab_overview, tab_predict, tab_forecast, tab_insights = st.tabs(
    ["📋 Overview", "🎯 No-Show Risk Predictor", "📈 Demand Forecasting", "📊 Insights"]
)


# ------------------------------------------------------------------
# TAB 1 — Overview
# ------------------------------------------------------------------
with tab_overview:
    st.subheader("At a glance")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total appointments (training data)", f"{stats['total_appointments']:,}")
    c2.metric("Historical no-show rate", f"{stats['no_show_rate']*100:.1f}%")
    c3.metric("Specialties covered", stats["n_specialties"])
    c4.metric("Clinic locations", stats["n_places"])

    st.markdown(f"Data period: **{stats['date_min']}** to **{stats['date_max']}**")

    st.divider()
    st.markdown("""
    **How to use this dashboard**
    - **No-Show Risk Predictor** — enter a patient's details before their visit to get a risk score,
      so staff can prioritize reminder calls or overbook high-risk slots.
    - **Demand Forecasting** — see projected appointment volume for the days ahead, optionally
      broken down by specialty, for staffing and scheduling decisions.
    - **Insights** — the key patterns behind both models (which factors drive no-shows and demand).
    """)


# ------------------------------------------------------------------
# TAB 2 — No-Show Risk Predictor
# ------------------------------------------------------------------
with tab_predict:
    st.subheader("Predict no-show risk for a scheduled appointment")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Patient**")
            age = st.number_input("Age", min_value=0, max_value=110, value=35)
            gender = st.selectbox("Gender", options=list(le_dict["gender"].classes_))
            disability = st.selectbox("Disability", options=list(le_dict["disability"].classes_))
            needs_companion = st.checkbox("Needs a companion")
            scholarship = st.checkbox("Has scholarship / social welfare benefit")

        with col2:
            st.markdown("**Appointment**")
            appt_date = st.date_input("Appointment date", value=dt.date.today())
            appointment_time = st.slider("Appointment hour (0–23)", 0, 23, 10)
            appointment_shift = st.selectbox("Shift", options=list(le_dict["appointment_shift"].classes_))
            specialty = st.selectbox("Specialty", options=list(le_dict["specialty"].classes_))
            place = st.selectbox("Clinic location", options=list(le_dict["place"].classes_))
            sms_received = st.checkbox("SMS reminder will be sent", value=True)

        with col3:
            st.markdown("**Health conditions**")
            hipertension = st.checkbox("Hypertension")
            diabetes = st.checkbox("Diabetes")
            alcoholism = st.checkbox("Alcoholism")
            handcap = st.checkbox("Handicap")
            st.markdown("**Weather forecast for appointment day**")
            avg_temp = st.number_input("Average temperature (°C)", value=22.0)
            max_temp = st.number_input("Max temperature (°C)", value=27.0)
            avg_rain = st.number_input("Average rainfall (mm)", value=0.0, min_value=0.0)
            max_rain = st.number_input("Max rainfall (mm)", value=0.0, min_value=0.0)
            rain_intensity = st.selectbox("Rain intensity", options=list(le_dict["rain_intensity"].classes_))
            heat_intensity = st.selectbox("Heat intensity", options=list(le_dict["heat_intensity"].classes_))
            rainy_day_before = st.checkbox("Rain the day before")
            storm_day_before = st.checkbox("Storm the day before")

        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        appt_date_ts = pd.Timestamp(appt_date)

        raw = {
            "specialty": specialty,
            "appointment_time": appointment_time,
            "gender": gender,
            "disability": disability,
            "place": place,
            "appointment_shift": appointment_shift,
            "age": age,
            "under_12_years_old": int(age < 12),
            "over_60_years_old": int(age > 60),
            "patient_needs_companion": int(needs_companion),
            "average_temp_day": avg_temp,
            "average_rain_day": avg_rain,
            "max_temp_day": max_temp,
            "max_rain_day": max_rain,
            "rainy_day_before": int(rainy_day_before),
            "storm_day_before": int(storm_day_before),
            "rain_intensity": rain_intensity,
            "heat_intensity": heat_intensity,
            "Hipertension": int(hipertension),
            "Diabetes": int(diabetes),
            "Alcoholism": int(alcoholism),
            "Handcap": int(handcap),
            "Scholarship": int(scholarship),
            "SMS_received": int(sms_received),
            "year": appt_date_ts.year,
            "month": appt_date_ts.month,
            "day": appt_date_ts.day,
            "weekday": appt_date_ts.dayofweek,
            "week": int(appt_date_ts.isocalendar().week),
        }

        row = pd.DataFrame([raw])

        # Apply the same label encoders used at training time
        for col, le in le_dict.items():
            if col in row.columns:
                val = row.at[0, col]
                if val in le.classes_:
                    row[col] = le.transform([val])
                else:
                    # unseen category -> fall back to the most frequent training class
                    row[col] = le.transform([le.classes_[0]])

        # Match training column order exactly; fill anything missing with 0
        for col in clf_features:
            if col not in row.columns:
                row[col] = 0
        row = row[clf_features]

        risk_score = float(clf.predict_proba(row)[0, 1])

        if risk_score < 0.3:
            risk_label, css_class = "Low risk", "risk-low"
        elif risk_score < 0.6:
            risk_label, css_class = "Medium risk", "risk-medium"
        else:
            risk_label, css_class = "High risk", "risk-high"

        st.divider()
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown("### Result")
            st.markdown(f"<h2 class='{css_class}'>{risk_label}</h2>", unsafe_allow_html=True)
            st.metric("Predicted no-show probability", f"{risk_score*100:.1f}%")
            if risk_score >= 0.3:
                st.info("Consider a reminder call, SMS follow-up, or flexible overbooking for this slot.")
        with r2:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2E5EAA"},
                    "steps": [
                        {"range": [0, 30], "color": "#DFF3E3"},
                        {"range": [30, 60], "color": "#FDF1DA"},
                        {"range": [60, 100], "color": "#FBDEDB"},
                    ],
                },
                title={"text": "No-Show Risk"},
            ))
            gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(gauge, use_container_width=True)

        with st.expander("What drives no-show risk in general?"):
            fi = tables["feature_importance"].head(10)
            fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                         title="Top 10 global risk factors (model-wide, not patient-specific)")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# TAB 3 — Demand Forecasting
# ------------------------------------------------------------------
with tab_forecast:
    st.subheader("Forecast upcoming appointment demand")

    last_hist_date = daily["appointment_date_continuous"].max().date()
    NEAR_TERM_LIMIT = 60  # days past last_hist_date where recursive (lag-based) forecasting stays reliable

    st.markdown("#### Pick a date")
    d1, d2 = st.columns([1, 1])
    with d1:
        target_date = st.date_input(
            "Predict appointment volume for this date",
            value=dt.date.today() + dt.timedelta(days=7),
            min_value=last_hist_date + dt.timedelta(days=1),
            max_value=dt.date.today() + dt.timedelta(days=730),
            help="Any real future date can be selected. The forecasting method used "
                 "automatically adapts based on how far the date is from the training data.",
        )
    with d2:
        specialty_options = ["All specialties"] + tables["specialty_share"]["specialty"].tolist()
        specialty_filter = st.selectbox("Specialty filter", options=specialty_options)

    horizon = (target_date - last_hist_date).days
    near_term = horizon <= NEAR_TERM_LIMIT

    if near_term:
        st.info(
            f"**Recursive model mode** ({horizon} days past the training data's last date, "
            f"{last_hist_date}). The trained regression model predicts day-by-day, feeding each "
            "prediction forward as the next day's input — the intended, most reliable use of this model."
        )
    else:
        st.warning(
            f"**Seasonal estimate mode** — {target_date} is {horizon} days past the training "
            f"data's last date ({last_hist_date}), too far for reliable day-by-day recursive "
            "forecasting (errors compound over that many steps). Instead, this estimate uses the "
            "trained model with typical day-of-week / seasonal demand patterns from history as its "
            "inputs, rather than a chained prediction sequence. It reflects the expected *typical* "
            "load for that kind of day, not a precise day-specific forecast."
        )

    def recursive_forecast(daily_df, model, feature_list, horizon):
        """Reliable for short horizons: chains each day's prediction into the next day's lag inputs."""
        history = daily_df.sort_values("appointment_date_continuous").reset_index(drop=True).copy()
        last_date = history["appointment_date_continuous"].max()

        trail = history.tail(30)
        context_defaults = {
            "sms_received_rate": trail["sms_received_rate"].mean(),
            "avg_age": trail["avg_age"].mean(),
            "hypertension_rate": trail["hypertension_rate"].mean(),
            "diabetes_rate": trail["diabetes_rate"].mean(),
            "avg_temp": trail["avg_temp"].mean(),
            "avg_rain": trail["avg_rain"].mean(),
        }

        counts = history["appointment_count"].tolist()
        forecasts = []

        for i in range(horizon):
            next_date = last_date + pd.Timedelta(days=i + 1)
            feat = {
                "dayofweek": next_date.dayofweek,
                "month": next_date.month,
                "quarter": next_date.quarter,
                "day": next_date.day,
                "is_weekend": int(next_date.dayofweek in [5, 6]),
                **context_defaults,
                "lag_1": counts[-1],
                "lag_7": counts[-7] if len(counts) >= 7 else counts[0],
                "lag_14": counts[-14] if len(counts) >= 14 else counts[0],
                "rolling_mean_7": np.mean(counts[-7:]),
                "rolling_mean_14": np.mean(counts[-14:]) if len(counts) >= 14 else np.mean(counts),
            }
            row = pd.DataFrame([feat])[feature_list]
            pred = max(0, float(model.predict(row)[0]))
            counts.append(pred)
            forecasts.append({"date": next_date, "predicted_appointments": pred})

        return pd.DataFrame(forecasts)

    def seasonal_forecast(daily_df, model, feature_list, target_date):
        """For dates far beyond the training data: uses day-of-week historical averages instead
        of a long chain of self-generated predictions, so errors can't compound."""
        history = daily_df.sort_values("appointment_date_continuous").reset_index(drop=True).copy()
        target_ts = pd.Timestamp(target_date)

        weekday_avg = history.groupby(
            history["appointment_date_continuous"].dt.dayofweek
        )["appointment_count"].mean()
        overall_avg = history["appointment_count"].mean()

        context = {
            "sms_received_rate": history["sms_received_rate"].mean(),
            "avg_age": history["avg_age"].mean(),
            "hypertension_rate": history["hypertension_rate"].mean(),
            "diabetes_rate": history["diabetes_rate"].mean(),
            "avg_temp": history["avg_temp"].mean(),
            "avg_rain": history["avg_rain"].mean(),
        }

        prev_day_wd = (target_ts - pd.Timedelta(days=1)).dayofweek
        feat = {
            "dayofweek": target_ts.dayofweek,
            "month": target_ts.month,
            "quarter": target_ts.quarter,
            "day": target_ts.day,
            "is_weekend": int(target_ts.dayofweek in [5, 6]),
            **context,
            "lag_1": weekday_avg.get(prev_day_wd, overall_avg),
            "lag_7": weekday_avg.get(target_ts.dayofweek, overall_avg),
            "lag_14": weekday_avg.get(target_ts.dayofweek, overall_avg),
            "rolling_mean_7": overall_avg,
            "rolling_mean_14": overall_avg,
        }
        row = pd.DataFrame([feat])[feature_list]
        pred = max(0, float(model.predict(row)[0]))

        # Build a light surrounding week (same method, day-by-day) purely for chart context
        window = []
        for offset in range(-3, 4):
            d = target_ts + pd.Timedelta(days=offset)
            wd = d.dayofweek
            f2 = {
                "dayofweek": wd, "month": d.month, "quarter": d.quarter, "day": d.day,
                "is_weekend": int(wd in [5, 6]), **context,
                "lag_1": weekday_avg.get((d - pd.Timedelta(days=1)).dayofweek, overall_avg),
                "lag_7": weekday_avg.get(wd, overall_avg),
                "lag_14": weekday_avg.get(wd, overall_avg),
                "rolling_mean_7": overall_avg, "rolling_mean_14": overall_avg,
            }
            r2 = pd.DataFrame([f2])[feature_list]
            p2 = max(0, float(model.predict(r2)[0]))
            window.append({"date": d, "predicted_appointments": p2})

        return pred, pd.DataFrame(window)

    if near_term:
        forecast_df = recursive_forecast(daily, reg, reg_features, horizon)
        target_prediction = forecast_df.iloc[-1]["predicted_appointments"]
        chart_forecast_df = forecast_df
    else:
        target_prediction, chart_forecast_df = seasonal_forecast(daily, reg, reg_features, target_date)

    if specialty_filter != "All specialties":
        share = tables["specialty_share"].set_index("specialty").loc[specialty_filter, "share"]
        target_prediction = target_prediction * share
        chart_forecast_df = chart_forecast_df.copy()
        chart_forecast_df["predicted_appointments"] = chart_forecast_df["predicted_appointments"] * share
        st.caption(
            f"Showing an estimated split for **{specialty_filter}** "
            f"(≈{share*100:.1f}% of historical volume). The underlying model forecasts "
            "total daily demand across all specialties; this is a proportional estimate, "
            "not a specialty-specific model."
        )

    # --- Headline answer: the specific date the user picked ---
    st.divider()
    a1, a2, a3 = st.columns([1.3, 1, 1])
    with a1:
        st.markdown(f"### {target_date.strftime('%A, %B %d, %Y')}")
        label = f"Predicted appointments{'' if specialty_filter == 'All specialties' else f' — {specialty_filter}'}"
        st.metric(label, f"{target_prediction:.0f}")
    with a2:
        st.metric("Days past training data", horizon)
    with a3:
        st.metric("Mode", "Recursive model" if near_term else "Seasonal estimate")

    st.divider()

    hist_tail = daily[["appointment_date_continuous", "appointment_count"]].tail(60).rename(
        columns={"appointment_date_continuous": "date", "appointment_count": "predicted_appointments"}
    )
    hist_tail["type"] = "Actual (last 60 days)"
    forecast_df_plot = chart_forecast_df.copy()
    forecast_df_plot["type"] = "Forecast" if near_term else "Seasonal pattern (±3 days shown)"
    combined = pd.concat([hist_tail, forecast_df_plot], ignore_index=True)

    fig = px.line(combined, x="date", y="predicted_appointments", color="type",
                   markers=True, title="Recent History + Forecast")
    fig.add_scatter(
        x=[pd.Timestamp(target_date)], y=[target_prediction],
        mode="markers", marker=dict(size=14, symbol="star", color="#D93025"),
        name="Selected date",
    )
    fig.update_layout(yaxis_title="Appointments", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    if near_term:
        st.markdown("##### Over the full forecast window")
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg. forecasted / day", f"{forecast_df['predicted_appointments'].mean():.0f}")
        c2.metric("Peak day", forecast_df.loc[forecast_df['predicted_appointments'].idxmax(), 'date'].strftime("%b %d"))
        c3.metric("Lowest day", forecast_df.loc[forecast_df['predicted_appointments'].idxmin(), 'date'].strftime("%b %d"))

        with st.expander("View forecast table"):
            display_df = forecast_df.copy()
            display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d (%a)")
            display_df["predicted_appointments"] = display_df["predicted_appointments"].round(1)
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ------------------------------------------------------------------
# TAB 4 — Insights
# ------------------------------------------------------------------
with tab_insights:
    st.subheader("Key patterns behind the models")

    i1, i2 = st.columns(2)
    with i1:
        fig = px.bar(tables["specialty"], x="no_show_rate", y="specialty", orientation="h",
                     title="No-Show Rate by Specialty")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(tables["rain"], x="rain_intensity", y="no_show_rate",
                     title="No-Show Rate by Rain Intensity")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with i2:
        fig = px.bar(tables["shift"], x="appointment_shift", y="no_show_rate",
                     title="No-Show Rate by Appointment Shift")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(tables["heat"], x="heat_intensity", y="no_show_rate",
                     title="No-Show Rate by Heat Intensity")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(tables["gender"], x="gender", y="no_show_rate", title="No-Show Rate by Gender")
    fig.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    fig = px.line(daily, x="appointment_date_continuous", y="appointment_count",
                   title="Historical Daily Appointment Volume")
    fig.update_layout(xaxis_title="Date", yaxis_title="Appointments")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    fi = tables["feature_importance"].head(15)
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 title="Top 15 Feature Importances — No-Show Classifier")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
