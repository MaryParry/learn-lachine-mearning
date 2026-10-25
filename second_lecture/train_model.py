"""
Model training, preprocessing, and evaluation script for Homework 1:
Regression Analysis with FastAPI (CarPrice_Assignment.csv).

Compares:
1. In-Class Baseline Model (from Untitled0.ipynb):
   - 4 features (fueltype, aspiration, enginelocation, curbweight) + unscaled OLS
2. Modified Approach A - Regularized Ridge Regression:
   - Brand name cleaning, correlation-based feature selection, One-Hot Encoding
     (drop_first=True), StandardScaler, and L2-regularized Ridge(alpha=5.0)
3. Modified Approach B - PCA Dimensionality Reduction Regression:
   - StandardScaler + Principal Component Analysis (PCA retaining 95% variance)
     to eliminate multicollinearity + Linear Regression
"""

from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "CarPrice_Assignment.csv"
MODEL_PATH = BASE_DIR / "car_price_model.joblib"

BRAND_CORRECTIONS = {
    "maxda": "mazda",
    "porcshce": "porsche",
    "toyouta": "toyota",
    "vokswagen": "volkswagen",
    "vw": "volkswagen",
}


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R2, RMSE, and MAE regression metrics."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_baseline_lecture_model(df: pd.DataFrame) -> Dict[str, float]:
    """Replicate the exact baseline model from second_lecture/Untitled0.ipynb."""
    data = df.copy()
    data.drop(["car_ID", "symboling"], axis=1, inplace=True)
    data["CarName"] = data["CarName"].map(lambda x: str(x).split()[0])
    data["CarName"] = data["CarName"].replace(BRAND_CORRECTIONS)

    cat_columns = data.select_dtypes(include=["object", "str"]).columns
    encoder = LabelEncoder()
    for col in cat_columns:
        if len(data[col].unique()) <= 3:
            data[col] = encoder.fit_transform(data[col])

    data = pd.get_dummies(
        data,
        columns=data.select_dtypes(include=["object", "str"]).columns,
        drop_first=True,
        dtype=int,
    )

    selected_data = data[
        ["price", "fueltype", "aspiration", "enginelocation", "curbweight"]
    ]
    y = selected_data["price"].values
    X = selected_data.drop("price", axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    preds = baseline_model.predict(X_test)
    return evaluate_predictions(y_test, preds)


def train_models() -> Dict[str, Any]:
    """Train baseline, modified Ridge, and PCA regression models and save bundle."""
    df = pd.read_csv(CSV_PATH)

    # 1. Evaluate lecture baseline (Untitled0.ipynb)
    baseline_metrics = train_baseline_lecture_model(df)

    # 2. Preprocess dataset for our modified regression pipeline
    data = df.copy()
    data["brand"] = data["CarName"].map(lambda x: str(x).split()[0].lower())
    data["brand"] = data["brand"].replace(BRAND_CORRECTIONS)

    numeric_feature_cols = [
        "horsepower",
        "enginesize",
        "curbweight",
        "carwidth",
        "highwaympg",
    ]
    categorical_cols = [
        "brand",
        "fueltype",
        "aspiration",
        "carbody",
        "drivewheel",
    ]
    selected_features = categorical_cols[:1] + numeric_feature_cols + categorical_cols[1:]

    # Correlation matrix with target price
    corr_series = data[numeric_feature_cols + ["price"]].corr()["price"]
    correlation_matrix = {col: float(corr_series[col]) for col in corr_series.index}

    # One-hot encode categorical columns (drop_first=True to avoid dummy variable trap)
    data_encoded = pd.get_dummies(
        data[selected_features],
        columns=categorical_cols,
        drop_first=True,
        dtype=float,
    )

    feature_names = list(data_encoded.columns)
    X = data_encoded.values
    y = data["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Modified Model A: Regularized Ridge Regression (handles multicollinearity)
    ridge_model = Ridge(alpha=5.0)
    ridge_model.fit(X_train_scaled, y_train)
    ridge_preds = ridge_model.predict(X_test_scaled)
    modified_metrics = evaluate_predictions(y_test, ridge_preds)

    # 4. Modified Model B: PCA Dimensionality Reduction + Linear Regression (Week 3 topic)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    pca_reg_model = LinearRegression()
    pca_reg_model.fit(X_train_pca, y_train)
    pca_preds = pca_reg_model.predict(X_test_pca)
    pca_metrics = evaluate_predictions(y_test, pca_preds)
    pca_metrics["n_components"] = int(pca.n_components_)
    pca_metrics["explained_variance_ratio_sum"] = float(
        np.sum(pca.explained_variance_ratio_)
    )

    coefficients = {
        name: float(weight)
        for name, weight in zip(feature_names, ridge_model.coef_)
    }
    intercept = float(ridge_model.intercept_)

    bundle: Dict[str, Any] = {
        "model": ridge_model,
        "pca": pca,
        "pca_model": pca_reg_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "categorical_cols": categorical_cols,
        "numeric_feature_cols": numeric_feature_cols,
        "brands": sorted(data["brand"].unique().tolist()),
        "carbodies": sorted(data["carbody"].unique().tolist()),
        "drivewheels": sorted(data["drivewheel"].unique().tolist()),
        "fueltypes": sorted(data["fueltype"].unique().tolist()),
        "aspirations": sorted(data["aspiration"].unique().tolist()),
        "feature_defaults": {
            col: float(data[col].median()) for col in numeric_feature_cols
        },
        "baseline_metrics": baseline_metrics,
        "modified_metrics": modified_metrics,
        "pca_metrics": pca_metrics,
        "coefficients": coefficients,
        "intercept": intercept,
        "correlation_matrix": correlation_matrix,
    }

    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    trained_bundle = train_models()
    b_m = trained_bundle["baseline_metrics"]
    m_m = trained_bundle["modified_metrics"]
    p_m = trained_bundle["pca_metrics"]

    print("=" * 65)
    print(" HOMEWORK 1: CAR PRICE REGRESSION TRAINING SUMMARY")
    print("=" * 65)
    print(f"Saved model bundle to: {MODEL_PATH}")
    print("-" * 65)
    print(
        f"1. Lecture Baseline (4-Feature OLS):   R² = {b_m['r2']:.4f} | RMSE = ${b_m['rmse']:,.2f} | MAE = ${b_m['mae']:,.2f}"
    )
    print(
        f"2. Modified Ridge Regression (Scaled): R² = {m_m['r2']:.4f} | RMSE = ${m_m['rmse']:,.2f} | MAE = ${m_m['mae']:,.2f}"
    )
    print(
        f"3. PCA Regression ({p_m['n_components']} comps, {p_m['explained_variance_ratio_sum']*100:.1f}% var): "
        f"R² = {p_m['r2']:.4f} | RMSE = ${p_m['rmse']:,.2f} | MAE = ${p_m['mae']:,.2f}"
    )
    print("=" * 65)
