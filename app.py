"""Flask web app for Employee Attrition Prediction."""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).parent
app = Flask(__name__)
model = joblib.load(BASE / "model" / "best_model.joblib")
meta = json.loads((BASE / "model" / "metrics.json").read_text())

FEATURES = meta["numeric_features"] + list(meta["categorical_features"])
LABELS = {c: c.replace("_", " ") for c in FEATURES}
best = max(meta["results"], key=lambda r: r["cv_roc_auc"])
# If even the best model barely beats a coin flip, tell the user.
LOW_SIGNAL = best["roc_auc"] < 0.6 and best["cv_roc_auc"] < 0.6


def risk_band(p):
    return "High" if p >= 0.6 else "Medium" if p >= 0.35 else "Low"


@app.route("/")
def index():
    return render_template("index.html", meta=meta, labels=LABELS, low_signal=LOW_SIGNAL)


@app.route("/models")
def models():
    return render_template("models.html", meta=meta, low_signal=LOW_SIGNAL)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()
    row, errors = {}, []
    for col in meta["numeric_features"]:
        try:
            row[col] = float(data[col])
        except (KeyError, ValueError, TypeError):
            errors.append(f"{LABELS[col]} must be a number")
    for col, options in meta["categorical_features"].items():
        if data.get(col) in options:
            row[col] = data[col]
        else:
            errors.append(f"{LABELS[col]} must be one of: {', '.join(options)}")
    if errors:
        return jsonify(error="; ".join(errors)), 400
    p = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
    return jsonify(probability=round(p, 4), risk=risk_band(p),
                   prediction="Likely to leave" if p >= 0.5 else "Likely to stay",
                   model=meta["selected_model"])


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(debug=True)
