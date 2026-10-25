"""
FastAPI Web Application for Homework 1:
Mini Project Assignment - Regression Analysis with FastAPI
Based on the CarPrice_Assignment.csv lecture dataset (Untitled0.ipynb).
"""

from pathlib import Path
from typing import Any, Dict
import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_model.joblib"

try:
    from second_lecture.train_model import train_models
except ImportError:
    from train_model import train_models

if not MODEL_PATH.exists():
    bundle = train_models()
else:
    bundle = joblib.load(MODEL_PATH)
    if "pca_metrics" not in bundle:
        bundle = train_models()

model = bundle["model"]
pca = bundle.get("pca")
pca_model = bundle.get("pca_model")
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]
brands = bundle["brands"]
carbodies = bundle["carbodies"]
drivewheels = bundle["drivewheels"]
fueltypes = bundle["fueltypes"]
aspirations = bundle["aspirations"]
feature_defaults = bundle["feature_defaults"]
baseline_metrics = bundle["baseline_metrics"]
modified_metrics = bundle["modified_metrics"]
pca_metrics = bundle["pca_metrics"]
coefficients = bundle["coefficients"]
intercept = bundle["intercept"]
correlation_matrix = bundle["correlation_matrix"]

app = FastAPI(
    title="Car Price Regression Analysis API",
    description=(
        "FastAPI Web Application supporting multivariable regression analysis, "
        "regularized Ridge regression, PCA dimensionality reduction, and car price prediction "
        "based on the CarPrice_Assignment lecture dataset."
    ),
    version="1.0.0",
)


class CarFeaturesInput(BaseModel):
    brand: str = Field(
        default="toyota",
        description="Car manufacturer brand (e.g., 'toyota', 'bmw', 'mazda')",
    )
    horsepower: float = Field(
        default=95.0, ge=40, le=500, description="Engine horsepower (hp)"
    )
    enginesize: float = Field(
        default=120.0, ge=50, le=500, description="Engine displacement size"
    )
    curbweight: float = Field(
        default=2414.0, ge=1000, le=6000, description="Vehicle curb weight (lbs)"
    )
    carwidth: float = Field(
        default=65.5, ge=50, le=90, description="Vehicle width (inches)"
    )
    highwaympg: float = Field(
        default=30.0, ge=10, le=80, description="Highway fuel economy (MPG)"
    )
    fueltype: str = Field(default="gas", description="Fuel type ('gas' or 'diesel')")
    aspiration: str = Field(
        default="std", description="Engine aspiration ('std' or 'turbo')"
    )
    carbody: str = Field(
        default="sedan",
        description="Body type ('sedan', 'hatchback', 'wagon', 'hardtop', 'convertible')",
    )
    drivewheel: str = Field(
        default="fwd", description="Drive wheel ('fwd', 'rwd', '4wd')"
    )


def compute_prediction(data_dict: Dict[str, Any]) -> Dict[str, float]:
    """Transform input specifications using the trained preprocessing pipeline and predict price."""
    brand_clean = str(data_dict.get("brand", "toyota")).lower().strip()
    brand_clean = {
        "maxda": "mazda",
        "porcshce": "porsche",
        "toyouta": "toyota",
        "vokswagen": "volkswagen",
        "vw": "volkswagen",
    }.get(brand_clean, brand_clean)

    row = {
        "brand": brand_clean,
        "horsepower": float(
            data_dict.get("horsepower", feature_defaults["horsepower"])
        ),
        "enginesize": float(
            data_dict.get("enginesize", feature_defaults["enginesize"])
        ),
        "curbweight": float(
            data_dict.get("curbweight", feature_defaults["curbweight"])
        ),
        "carwidth": float(data_dict.get("carwidth", feature_defaults["carwidth"])),
        "highwaympg": float(
            data_dict.get("highwaympg", feature_defaults["highwaympg"])
        ),
        "fueltype": str(data_dict.get("fueltype", "gas")).lower().strip(),
        "aspiration": str(data_dict.get("aspiration", "std")).lower().strip(),
        "carbody": str(data_dict.get("carbody", "sedan")).lower().strip(),
        "drivewheel": str(data_dict.get("drivewheel", "fwd")).lower().strip(),
    }

    df_single = pd.DataFrame([row])
    df_encoded = pd.get_dummies(
        df_single,
        columns=["brand", "fueltype", "aspiration", "carbody", "drivewheel"],
        dtype=float,
    )

    df_aligned = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for col in df_encoded.columns:
        if col in df_aligned.columns:
            df_aligned[col] = df_encoded[col].values

    scaled_features = scaler.transform(df_aligned.values)
    ridge_val = float(model.predict(scaled_features)[0])

    pca_val = ridge_val
    if pca is not None and pca_model is not None:
        pca_features = pca.transform(scaled_features)
        pca_val = float(pca_model.predict(pca_features)[0])

    return {
        "ridge_price": max(ridge_val, 3000.0),
        "pca_price": max(pca_val, 3000.0),
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": True}


@app.get("/api/options")
def get_options():
    return {
        "brands": brands,
        "carbodies": carbodies,
        "drivewheels": drivewheels,
        "fueltypes": fueltypes,
        "aspirations": aspirations,
        "feature_defaults": feature_defaults,
    }


@app.get("/api/analysis")
def get_analysis():
    sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
    return {
        "dataset": "CarPrice_Assignment.csv",
        "baseline_model": {
            "description": "In-Class OLS Regression (4 features: fueltype, aspiration, enginelocation, curbweight)",
            "metrics": baseline_metrics,
        },
        "modified_model": {
            "description": "Modified Ridge Regression (L2 alpha=5.0) with Correlation Feature Selection + One-Hot Encoding + StandardScaler",
            "metrics": modified_metrics,
            "intercept": intercept,
            "top_coefficients": dict(sorted_coefs[:15]),
        },
        "pca_model": {
            "description": f"PCA Dimensionality Reduction ({pca_metrics['n_components']} components, {pca_metrics['explained_variance_ratio_sum'] * 100:.1f}% variance) + Linear Regression",
            "metrics": pca_metrics,
        },
        "correlation_with_price": correlation_matrix,
    }


@app.post("/api/predict")
def predict_price(payload: CarFeaturesInput):
    preds = compute_prediction(payload.model_dump())
    pred = preds["ridge_price"]
    pca_pred = preds["pca_price"]

    if pred < 10000:
        segment = "Economy / Budget"
    elif pred < 18000:
        segment = "Mid-Range / Consumer"
    elif pred < 28000:
        segment = "Entry Luxury / Sport"
    else:
        segment = "High-End / Luxury"

    return {
        "predicted_price": round(pred, 2),
        "formatted_price": f"${pred:,.2f}",
        "pca_predicted_price": round(pca_pred, 2),
        "pca_formatted_price": f"${pca_pred:,.2f}",
        "market_segment": segment,
        "inputs": payload.model_dump(),
    }


@app.get("/", response_class=HTMLResponse)
def index_page():
    brand_options = "".join(
        f'<option value="{b}" {"selected" if b == "toyota" else ""}>{b.title()}</option>'
        for b in brands
    )
    body_options = "".join(
        f'<option value="{b}" {"selected" if b == "sedan" else ""}>{b.title()}</option>'
        for b in carbodies
    )
    drive_options = "".join(
        f'<option value="{d}" {"selected" if d == "fwd" else ""}>{d.upper()}</option>'
        for d in drivewheels
    )
    fuel_options = "".join(
        f'<option value="{f}" {"selected" if f == "gas" else ""}>{f.title()}</option>'
        for f in fueltypes
    )
    aspiration_options = "".join(
        f'<option value="{a}" {"selected" if a == "std" else ""}>{a.upper()}</option>'
        for a in aspirations
    )

    sorted_coefs = sorted(coefficients.items(), key=lambda x: x[1], reverse=True)
    top_pos = sorted_coefs[:6]
    top_neg = sorted_coefs[-5:]

    coef_rows = ""
    for name, weight in top_pos + top_neg:
        sign = "+" if weight >= 0 else ""
        coef_rows += f"""
        <tr>
            <td><code>{name}</code></td>
            <td style="text-align: right;"><code>{sign}{weight:.2f}</code></td>
        </tr>
        """

    corr_rows = ""
    for feat, corr_val in sorted(
        correlation_matrix.items(), key=lambda x: abs(x[1]), reverse=True
    ):
        if feat == "price":
            continue
        corr_rows += f"""
        <tr>
            <td>{feat}</td>
            <td style="text-align: right;"><code>{corr_val:+.4f}</code></td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Car Price Regression Analysis</title>
    <style>
        body {{
            font-family: sans-serif;
            max-width: 720px;
            margin: 28px auto;
            padding: 0 16px;
            line-height: 1.4;
            color: #111;
        }}
        h1 {{
            font-size: 1.4rem;
            margin-bottom: 4px;
        }}
        h2 {{
            font-size: 1.15rem;
            margin-top: 24px;
            margin-bottom: 8px;
            border-bottom: 1px solid #ccc;
            padding-bottom: 4px;
        }}
        .form-row {{
            margin-bottom: 8px;
        }}
        label {{
            display: inline-block;
            width: 120px;
            font-size: 0.9rem;
        }}
        input, select {{
            padding: 4px;
            width: 180px;
        }}
        button {{
            padding: 6px 14px;
            cursor: pointer;
            margin-top: 8px;
        }}
        .result {{
            background: #f9f9f9;
            border: 1px solid #ccc;
            padding: 10px 14px;
            margin: 14px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0 16px;
            font-size: 0.88rem;
        }}
        th, td {{
            border: 1px solid #ccc;
            padding: 5px 8px;
            text-align: left;
        }}
        th {{
            background: #eee;
        }}
        ul {{
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 3px;
        }}
    </style>
</head>
<body>
    <h1>Car Price Regression Analysis</h1>
    <p>FastAPI web application for regression analysis using <code>CarPrice_Assignment.csv</code>.</p>

    <h2>1. Car Price Estimator</h2>
    <form id="calcForm" onsubmit="event.preventDefault(); predict();">
        <div class="form-row">
            <label for="brand">Brand:</label>
            <select id="brand">{brand_options}</select>

            <label for="carbody" style="margin-left: 16px;">Body Style:</label>
            <select id="carbody">{body_options}</select>
        </div>

        <div class="form-row">
            <label for="horsepower">Horsepower:</label>
            <input type="number" id="horsepower" value="95" min="40" max="400" required>

            <label for="enginesize" style="margin-left: 16px;">Engine Size:</label>
            <input type="number" id="enginesize" value="120" min="50" max="400" required>
        </div>

        <div class="form-row">
            <label for="curbweight">Curb Weight:</label>
            <input type="number" id="curbweight" value="2414" min="1200" max="5000" required>

            <label for="carwidth" style="margin-left: 16px;">Width (in):</label>
            <input type="number" step="0.1" id="carwidth" value="65.5" min="55" max="85" required>
        </div>

        <div class="form-row">
            <label for="highwaympg">Highway MPG:</label>
            <input type="number" id="highwaympg" value="30" min="12" max="60" required>

            <label for="drivewheel" style="margin-left: 16px;">Drive Wheel:</label>
            <select id="drivewheel">{drive_options}</select>
        </div>

        <div class="form-row">
            <label for="fueltype">Fuel Type:</label>
            <select id="fueltype">{fuel_options}</select>

            <label for="aspiration" style="margin-left: 16px;">Aspiration:</label>
            <select id="aspiration">{aspiration_options}</select>
        </div>

        <button type="submit">Calculate Predicted Price</button>
    </form>

    <div class="result">
        <p><strong>Predicted Price (Ridge L2):</strong> <span id="predictedPrice" style="font-weight: bold; font-size: 1.15rem;">$10,480.00</span></p>
        <p><strong>PCA Regression Estimate:</strong> <span id="pcaPredictedPrice">$10,390.00</span></p>
        <p><strong>Market Category:</strong> <span id="segmentBadge">Mid-Range / Consumer</span></p>
    </div>

    <h2>2. Model Performance Comparison</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>R²</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Details</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>In-Class Baseline</td>
                <td>{baseline_metrics['r2']:.4f}</td>
                <td>${baseline_metrics['rmse']:,.2f}</td>
                <td>${baseline_metrics['mae']:,.2f}</td>
                <td>OLS (4 unscaled features from lecture)</td>
            </tr>
            <tr>
                <td>Modified Ridge Model</td>
                <td>{modified_metrics['r2']:.4f}</td>
                <td>${modified_metrics['rmse']:,.2f}</td>
                <td>${modified_metrics['mae']:,.2f}</td>
                <td>StandardScaler + Ridge (&alpha; = 5.0)</td>
            </tr>
            <tr>
                <td>PCA Regression</td>
                <td>{pca_metrics['r2']:.4f}</td>
                <td>${pca_metrics['rmse']:,.2f}</td>
                <td>${pca_metrics['mae']:,.2f}</td>
                <td>{pca_metrics['n_components']} components ({pca_metrics['explained_variance_ratio_sum'] * 100:.1f}% variance)</td>
            </tr>
        </tbody>
    </table>

    <p><strong>Preprocessing & Modifications:</strong></p>
    <ul>
        <li>Extracted brand names from <code>CarName</code> and fixed typos (e.g. <code>toyouta</code> &rarr; <code>toyota</code>).</li>
        <li>Selected highest-correlation features: enginesize, curbweight, horsepower, carwidth, highwaympg.</li>
        <li>One-hot encoded categorical variables with <code>drop_first=True</code>.</li>
        <li>Applied <code>StandardScaler</code> (&mu; = 0, &sigma; = 1) across numeric features.</li>
        <li>Applied L2 Ridge penalty and PCA to resolve multicollinearity.</li>
    </ul>

    <h2>3. Regression Coefficients (Weights)</h2>
    <p>Intercept ($w_0$): <strong>${intercept:+,.2f}</strong></p>
    <table>
        <thead>
            <tr>
                <th>Feature / Variable</th>
                <th style="text-align: right;">Standardized Weight</th>
            </tr>
        </thead>
        <tbody>
            {coef_rows}
        </tbody>
    </table>

    <h2>4. Correlations with Price</h2>
    <table>
        <thead>
            <tr>
                <th>Variable</th>
                <th style="text-align: right;">Pearson Correlation (r)</th>
            </tr>
        </thead>
        <tbody>
            {corr_rows}
        </tbody>
    </table>

    <p style="margin-top: 24px; font-size: 0.85rem; color: #555;">
        API Endpoints: <code>POST /api/predict</code> &bull; <code>GET /api/analysis</code> &bull; <a href="/docs">FastAPI Interactive Docs</a>
    </p>

<script>
    async function predict() {{
        const payload = {{
            brand: document.getElementById('brand').value,
            carbody: document.getElementById('carbody').value,
            horsepower: parseFloat(document.getElementById('horsepower').value),
            enginesize: parseFloat(document.getElementById('enginesize').value),
            curbweight: parseFloat(document.getElementById('curbweight').value),
            carwidth: parseFloat(document.getElementById('carwidth').value),
            highwaympg: parseFloat(document.getElementById('highwaympg').value),
            drivewheel: document.getElementById('drivewheel').value,
            fueltype: document.getElementById('fueltype').value,
            aspiration: document.getElementById('aspiration').value
        }};

        try {{
            const res = await fetch('/api/predict', {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify(payload)
            }});
            const data = await res.json();
            document.getElementById('predictedPrice').textContent = data.formatted_price;
            document.getElementById('pcaPredictedPrice').textContent = data.pca_formatted_price;
            document.getElementById('segmentBadge').textContent = data.market_segment;
        }} catch (err) {{
            console.error(err);
        }}
    }}

    window.addEventListener('DOMContentLoaded', predict);
</script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
