"""
Multivariable Linear Regression on CarPrice_Assignment dataset.
Demonstrates baseline multivariable regression vs modified regularized regression.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Locate dataset
BASE_DIR = Path(__file__).resolve().parent
csv_path = BASE_DIR / "CarPrice_Assignment.csv"

# 1. Load Data
df = pd.read_csv(csv_path)

# 2. Preprocess brand names (as done in lecture notebook)
data = df.copy()
data["brand"] = data["CarName"].map(lambda x: str(x).split()[0].lower())
data["brand"] = data["brand"].replace({
    "maxda": "mazda",
    "porcshce": "porsche",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
})

# 3. Select multivariable features
selected_features = [
    "brand",
    "horsepower",
    "enginesize",
    "curbweight",
    "carwidth",
    "highwaympg",
    "fueltype",
    "aspiration",
    "carbody",
    "drivewheel",
]

# 4. One-Hot Encode categorical variables
data_encoded = pd.get_dummies(
    data[selected_features],
    columns=["brand", "fueltype", "aspiration", "carbody", "drivewheel"],
    drop_first=True,
    dtype=float,
)

X = data_encoded
y = data["price"]

# 5. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 6. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Fit Models
# Baseline OLS
ols_model = LinearRegression()
ols_model.fit(X_train_scaled, y_train)
ols_preds = ols_model.predict(X_test_scaled)

# Modified Ridge Regression (Regularized to handle multicollinearity)
ridge_model = Ridge(alpha=5.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_preds = ridge_model.predict(X_test_scaled)

print("=" * 60)
print(" MULTIVARIABLE REGRESSION RESULTS (CarPrice_Assignment.csv)")
print("=" * 60)
print(f"Dataset Shape: {df.shape[0]} samples, {len(selected_features)} raw features")
print(f"Processed Features (after one-hot encoding): {X.shape[1]}")
print("-" * 60)
print("Standard Linear Regression (OLS):")
print(f"  R² Score: {r2_score(y_test, ols_preds):.4f}")
print(f"  RMSE:     ${np.sqrt(mean_squared_error(y_test, ols_preds)):,.2f}")
print(f"  MAE:      ${mean_absolute_error(y_test, ols_preds):,.2f}")
print("-" * 60)
print("Modified Model - Ridge Regression (L2 penalty):")
print(f"  R² Score: {r2_score(y_test, ridge_preds):.4f}")
print(f"  RMSE:     ${np.sqrt(mean_squared_error(y_test, ridge_preds)):,.2f}")
print(f"  MAE:      ${mean_absolute_error(y_test, ridge_preds):,.2f}")
print("=" * 60)
