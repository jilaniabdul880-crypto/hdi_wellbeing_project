"""
app.py
------
Flask web application for the "Comprehensive Measure of Well-Being"
project. Serves an interactive UI where a user enters the four UNDP
development indicators and receives:

  * The predicted HDI tier (Low / Medium / High / Very High) from the
    trained scikit-learn classifier, with class probabilities.
  * The exact numerical HDI score computed with the official UNDP
    geometric-mean formula, for transparency / comparison against the
    ML model's prediction.
"""

import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load trained artifacts once at startup
# ---------------------------------------------------------------------
model = joblib.load(os.path.join(MODELS_DIR, "hdi_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    METADATA = json.load(f)

FEATURES = METADATA["features"]
TIER_ORDER = METADATA["tier_order"]

LE_MIN, LE_MAX = 20, 85
MYS_MAX = 15
EYS_MAX = 18
GNI_MIN, GNI_MAX = 100, 75000

TIER_INFO = {
    "Low": {
        "color": "#E76F51",
        "blurb": "Substantial gaps in health, education, and income. "
                 "Targeted investment in these areas would meaningfully "
                 "raise human development outcomes.",
    },
    "Medium": {
        "color": "#F4A261",
        "blurb": "A developing profile with moderate outcomes across the "
                 "board. Focused policy gains in the weakest dimension "
                 "would move this profile toward the High tier.",
    },
    "High": {
        "color": "#2A9D8F",
        "blurb": "Strong performance across most dimensions, with room to "
                 "close remaining gaps in schooling or income to reach the "
                 "Very High tier.",
    },
    "Very High": {
        "color": "#264653",
        "blurb": "Among the most developed profiles: long healthy lives, "
                 "extensive education, and high income are all present "
                 "together.",
    },
}


def official_hdi(life_expectancy, mys, eys, gni):
    """Compute the exact UNDP-style HDI score for reference/comparison."""
    lei = (life_expectancy - LE_MIN) / (LE_MAX - LE_MIN)
    mysi = min(mys / MYS_MAX, 1)
    eysi = min(eys / EYS_MAX, 1)
    ei = (mysi + eysi) / 2
    gni_clamped = min(max(gni, GNI_MIN), GNI_MAX)
    ii = (np.log(gni_clamped) - np.log(GNI_MIN)) / (np.log(GNI_MAX) - np.log(GNI_MIN))

    lei = max(lei, 1e-6)
    ei = max(ei, 1e-6)
    ii = max(ii, 1e-6)
    score = (lei * ei * ii) ** (1 / 3)
    return round(float(np.clip(score, 0, 1)), 4)


def official_tier(score):
    if score >= 0.800:
        return "Very High"
    if score >= 0.700:
        return "High"
    if score >= 0.550:
        return "Medium"
    return "Low"


@app.route("/")
def index():
    return render_template(
        "index.html",
        ranges={
            "life_expectancy": (LE_MIN, LE_MAX),
            "mean_years_schooling": (0, MYS_MAX),
            "expected_years_schooling": (0, EYS_MAX),
            "gni_per_capita": (GNI_MIN, GNI_MAX),
        },
        model_name=METADATA["model_name"],
        test_accuracy=METADATA["test_accuracy"],
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        life_expectancy = float(payload["life_expectancy"])
        mys = float(payload["mean_years_schooling"])
        eys = float(payload["expected_years_schooling"])
        gni = float(payload["gni_per_capita"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid or missing input values."}), 400

    # Basic bounds validation
    bounds = {
        "life_expectancy": (LE_MIN, LE_MAX),
        "mean_years_schooling": (0, MYS_MAX),
        "expected_years_schooling": (0, EYS_MAX),
        "gni_per_capita": (GNI_MIN, GNI_MAX),
    }
    values = {
        "life_expectancy": life_expectancy,
        "mean_years_schooling": mys,
        "expected_years_schooling": eys,
        "gni_per_capita": gni,
    }
    for key, (lo, hi) in bounds.items():
        if not (lo <= values[key] <= hi):
            return jsonify({
                "error": f"'{key}' must be between {lo} and {hi} (got {values[key]})."
            }), 400

    X = np.array([[life_expectancy, mys, eys, gni]])
    X_scaled = scaler.transform(X)

    pred_idx = model.predict(X_scaled)[0]
    pred_tier = encoder.inverse_transform([pred_idx])[0]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0]
        for idx, p in enumerate(proba):
            tier_name = encoder.inverse_transform([idx])[0]
            probabilities[tier_name] = round(float(p), 4)
        probabilities = {t: probabilities.get(t, 0.0) for t in TIER_ORDER}

    hdi_score = official_hdi(life_expectancy, mys, eys, gni)
    formula_tier = official_tier(hdi_score)

    response = {
        "predicted_tier": pred_tier,
        "probabilities": probabilities,
        "hdi_score": hdi_score,
        "formula_tier": formula_tier,
        "tier_info": TIER_INFO[pred_tier],
        "tier_order": TIER_ORDER,
        "model_name": METADATA["model_name"],
    }
    return jsonify(response)


@app.route("/api/model-info")
def model_info():
    return jsonify(METADATA)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
