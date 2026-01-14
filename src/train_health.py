import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =================================================
# Utility: Convert NumPy types to JSON-safe types
# =================================================
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# =================================================
# Paths
# =================================================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_health_insurance_20_features.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# =================================================
# Load Dataset
# =================================================
df = pd.read_csv(DATA_PATH)

X = df.drop("monthly_premium_inr", axis=1)
y = df["monthly_premium_inr"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

# =================================================
# 1. Linear Regression
# =================================================
linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_pipeline.fit(X_train, y_train)
y_pred = linear_pipeline.predict(X_test)

results["linear_regression"] = {
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
}

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Premium")
plt.ylabel("Predicted Premium")
plt.title("Linear Regression: Actual vs Predicted")
plt.savefig(os.path.join(FIGURES_DIR, "pred_vs_actual_linear.png"))
plt.close()

# =================================================
# 2. Lasso Regression
# =================================================
lasso_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(max_iter=10000))
])

lasso_params = {"lasso__alpha": [0.1, 1, 10, 50, 100]}
lasso_grid = GridSearchCV(lasso_pipeline, lasso_params, cv=5)
lasso_grid.fit(X_train, y_train)

y_pred = lasso_grid.predict(X_test)

results["lasso_regression"] = {
    "best_alpha": lasso_grid.best_params_["lasso__alpha"],
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
}

lasso_coef = pd.DataFrame({
    "feature": X.columns,
    "coefficient": lasso_grid.best_estimator_.named_steps["lasso"].coef_
})
lasso_coef.to_csv(os.path.join(OUTPUTS_DIR, "lasso_coefficients.csv"), index=False)

# =================================================
# 3. Ridge Regression
# =================================================
ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge())
])

ridge_params = {"ridge__alpha": np.logspace(-2, 2, 10)}
ridge_grid = GridSearchCV(ridge_pipeline, ridge_params, cv=5)
ridge_grid.fit(X_train, y_train)

y_pred = ridge_grid.predict(X_test)

results["ridge_regression"] = {
    "best_alpha": ridge_grid.best_params_["ridge__alpha"],
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
}

ridge_coef = pd.DataFrame({
    "feature": X.columns,
    "coefficient": ridge_grid.best_estimator_.named_steps["ridge"].coef_
})
ridge_coef.to_csv(os.path.join(OUTPUTS_DIR, "ridge_coefficients.csv"), index=False)

# =================================================
# 4. Support Vector Regression (SVR)
# =================================================
svr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf"))
])

svr_params = {
    "svr__C": [100, 1000],
    "svr__epsilon": [10, 100],
    "svr__gamma": ["scale"]
}

svr_grid = GridSearchCV(svr_pipeline, svr_params, cv=3)
svr_grid.fit(X_train, y_train)

y_pred = svr_grid.predict(X_test)

results["svr"] = {
    "best_params": svr_grid.best_params_,
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
}

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Premium")
plt.ylabel("Predicted Premium")
plt.title("SVR: Actual vs Predicted")
plt.savefig(os.path.join(FIGURES_DIR, "pred_vs_actual_svr.png"))
plt.close()

# =================================================
# 5. PCA + Linear Regression
# =================================================
pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("model", LinearRegression())
])

pca_pipeline.fit(X_train, y_train)
y_pred = pca_pipeline.predict(X_test)

results["pca_linear_regression"] = {
    "n_components": pca_pipeline.named_steps["pca"].n_components_,
    "explained_variance_ratio_sum":
        pca_pipeline.named_steps["pca"].explained_variance_ratio_.sum(),
    "R2": r2_score(y_test, y_pred),
    "MAE": mean_absolute_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
}

plt.figure()
plt.plot(np.cumsum(pca_pipeline.named_steps["pca"].explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot")
plt.savefig(os.path.join(FIGURES_DIR, "pca_scree.png"))
plt.close()

# =================================================
# Save Results (JSON-safe)
# =================================================
results_clean = make_json_serializable(results)

with open(os.path.join(OUTPUTS_DIR, "metrics_summary.json"), "w") as f:
    json.dump(results_clean, f, indent=4)

print("Training completed successfully!")
print("Results saved in 'figures/' and 'outputs/'")
