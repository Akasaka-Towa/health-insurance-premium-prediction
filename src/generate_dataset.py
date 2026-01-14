import os
import numpy as np
import pandas as pd

np.random.seed(42)

# -------------------------------
# Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(DATA_DIR, "synthetic_health_insurance_20_features.csv")

# -------------------------------
# Number of samples
# -------------------------------
N = 1000  # You can increase to 5000 if needed

# -------------------------------
# Feature generation
# -------------------------------
data = {
    "age_years": np.random.randint(18, 70, N),
    "bmi": np.round(np.random.uniform(18, 40, N), 1),
    "is_smoker": np.random.binomial(1, 0.25, N),
    "is_male": np.random.binomial(1, 0.5, N),
    "children_count": np.random.randint(0, 5, N),
    "annual_income_lakh": np.round(np.random.uniform(3, 30, N), 1),
    "exercise_hours_per_week": np.round(np.random.uniform(0, 10, N), 1),
    "diet_quality_score": np.random.randint(1, 11, N),
    "chronic_conditions_count": np.random.randint(0, 4, N),
    "hospital_visits_last_year": np.random.randint(0, 6, N),
    "medication_cost_monthly": np.random.randint(0, 5000, N),
    "stress_level_index": np.random.randint(1, 11, N),
    "alcohol_units_per_week": np.random.randint(0, 15, N),
    "sleep_hours_per_day": np.round(np.random.uniform(4, 9, N), 1),
    "bp_systolic_mmHg": np.random.randint(110, 170, N),
    "cholesterol_mg_dl": np.random.randint(150, 300, N),
    "has_annual_health_checkup": np.random.binomial(1, 0.6, N),
    "family_history_score": np.random.randint(0, 11, N),
    "dental_visits_last_year": np.random.randint(0, 4, N),
    "screenings_completed": np.random.randint(0, 5, N),
}

df = pd.DataFrame(data)

# -------------------------------
# Target variable (premium logic)
# -------------------------------
premium = (
    2000
    + df["age_years"] * 45
    + df["bmi"] * 120
    + df["is_smoker"] * 5000
    + df["chronic_conditions_count"] * 2500
    + df["hospital_visits_last_year"] * 1200
    + df["medication_cost_monthly"] * 0.8
    + df["stress_level_index"] * 300
    - df["exercise_hours_per_week"] * 200
    - df["diet_quality_score"] * 150
    - df["has_annual_health_checkup"] * 800
    + np.random.normal(0, 1500, N)
)

df["monthly_premium_inr"] = premium.round(0).astype(int)

# -------------------------------
# Save CSV
# -------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("Synthetic dataset generated successfully!")
print(f"File saved at: {OUTPUT_PATH}")
print(f"Total samples: {len(df)}")

