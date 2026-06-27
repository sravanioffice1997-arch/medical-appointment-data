# 🏥 Medical Appointment Analytics

## 📌 Project Overview

Medical Appointment Analytics is a Machine Learning project developed to help healthcare organizations improve appointment management.

The project contains two machine learning modules:

* 🤖 **No-Show Prediction** – Predicts whether a patient is likely to miss an appointment.
* 📈 **Demand Forecasting** – Predicts the expected number of appointments for future dates.

Both models are integrated into a single interactive Streamlit dashboard.

---

# 📊 Dataset

* **Total Records:** 109,593
* **Features Used:** 29
* **Target Variable (Classification):** `no_show`
* **Target Variable (Regression):** Daily appointment count

The dataset includes patient information, appointment details, weather conditions, medical history, and appointment outcomes.

---

# 🔍 Exploratory Data Analysis (EDA)

Performed extensive EDA including:

* Missing value analysis
* Class distribution
* Age distribution
* Gender analysis
* Appointment shift analysis
* Weather analysis
* Correlation analysis
* Feature engineering

---

# ⚙️ Feature Engineering

Additional features created include:

### Classification

* Year
* Month
* Day
* Weekday
* Week

### Demand Forecasting

* Day of Week
* Month
* Quarter
* Day
* Lag 1
* Lag 7
* Lag 14
* Rolling Mean (7 days)
* Rolling Mean (14 days)

---

# 🤖 Machine Learning Models

## 1. No-Show Prediction

**Algorithm**

* XGBoost Classifier

**Evaluation Metrics**

| Metric   | Score     |
| -------- | --------- |
| F1 Score | **0.629** |
| ROC-AUC  | **0.783** |

---

## 2. Demand Forecasting

Three regression models were evaluated.

| Model                   |        MAE |       RMSE |        R² |
| ----------------------- | ---------: | ---------: | --------: |
| Linear Regression       |     161.46 |     241.16 |     0.278 |
| Random Forest Regressor | **135.46** | **230.57** | **0.340** |
| XGBoost Regressor       |     150.19 |     245.03 |     0.255 |

**Selected Model:** Random Forest Regressor

---

# 🖥️ Streamlit Dashboard

The application contains three pages:

### 🏠 Home

* Project overview
* Model performance metrics

### 🤖 No-Show Prediction

Predicts whether a patient is likely to attend or miss an appointment based on patient and appointment information.

### 📈 Demand Forecasting

Predicts the expected number of future appointments using historical demand features.

---


# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Streamlit
* Matplotlib
* Seaborn

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sravanioffice1997-arch/medical-appointment-data.git
```

Navigate to the project directory:

```bash
cd medical-appointment-data
```

Install dependencies:

```bash
pip install -r 06_requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```


## ⭐ If you found this project useful, consider giving the repository a star!
