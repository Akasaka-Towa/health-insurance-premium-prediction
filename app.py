import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ------------------------------------
# Load dataset & train model
# ------------------------------------
DATA_PATH = "data/synthetic_health_insurance_20_features.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("monthly_premium_inr", axis=1)
y = df["monthly_premium_inr"]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X, y)

# ------------------------------------
# Streamlit UI
# ------------------------------------
st.set_page_config(page_title="Health Insurance Premium Predictor")

st.title("🏥 Health Insurance Premium Predictor")
st.write("Enter your details to predict monthly insurance premium (INR)")

# ------------------------------------
# User Inputs
# ------------------------------------
age = st.slider("Age", 18, 70, 30)
bmi = st.slider("BMI", 18.0, 40.0, 25.0)
is_smoker = st.selectbox("Smoker", [0, 1])
is_male = st.selectbox("Gender (Male=1, Female=0)", [0, 1])
children = st.slider("Number of Children", 0, 4, 0)
income = st.slider("Annual Income (Lakh INR)", 3.0, 30.0, 10.0)
exercise = st.slider("Exercise Hours / Week", 0.0, 10.0, 3.0)
diet = st.slider("Diet Quality (1–10)", 1, 10, 6)
chronic = st.slider("Chronic Conditions", 0, 3, 0)
hospital_visits = st.slider("Hospital Visits Last Year", 0, 5, 1)
medicine_cost = st.slider("Monthly Medicine Cost (INR)", 0, 5000, 500)
stress = st.slider("Stress Level (1–10)", 1, 10, 5)
alcohol = st.slider("Alcohol Units / Week", 0, 15, 2)
sleep = st.slider("Sleep Hours / Day", 4.0, 9.0, 7.0)
bp = st.slider("Systolic BP", 110, 170, 120)
cholesterol = st.slider("Cholesterol (mg/dL)", 150, 300, 200)
checkup = st.selectbox("Annual Health Checkup", [0, 1])
family_history = st.slider("Family History Score", 0, 10, 3)
dental = st.slider("Dental Visits Last Year", 0, 3, 1)
screenings = st.slider("Preventive Screenings", 0, 4, 1)

# ------------------------------------
# Prediction
# ------------------------------------
input_data = pd.DataFrame([{
    "age_years": age,
    "bmi": bmi,
    "is_smoker": is_smoker,
    "is_male": is_male,
    "children_count": children,
    "annual_income_lakh": income,
    "exercise_hours_per_week": exercise,
    "diet_quality_score": diet,
    "chronic_conditions_count": chronic,
    "hospital_visits_last_year": hospital_visits,
    "medication_cost_monthly": medicine_cost,
    "stress_level_index": stress,
    "alcohol_units_per_week": alcohol,
    "sleep_hours_per_day": sleep,
    "bp_systolic_mmHg": bp,
    "cholesterol_mg_dl": cholesterol,
    "has_annual_health_checkup": checkup,
    "family_history_score": family_history,
    "dental_visits_last_year": dental,
    "screenings_completed": screenings
}])




if st.button("Predict Premium"):
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Monthly Premium: ₹ {int(prediction)}")
