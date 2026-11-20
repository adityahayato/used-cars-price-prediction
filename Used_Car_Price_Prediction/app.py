from pathlib import Path
import logging
import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
app = Flask(__name__)
app.config["SECRET_KEY"] = "used-car-prediction-local"
logging.basicConfig(level=logging.INFO)


def model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The model is not trained yet. Put data/car_data.csv in place and run: python train_model.py")
    return joblib.load(MODEL_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        bundle = model_bundle()
    except FileNotFoundError as error:
        return render_template("prediction.html", error=str(error), form=request.form)
    if request.method == "GET":
        return render_template("prediction.html", fields=bundle["features"], defaults=bundle.get("feature_defaults", {}))
    values = {}
    for feature in bundle["features"]:
        raw = request.form.get(feature, "").strip()
        if raw == "":
            return render_template("prediction.html", error=f"Please provide a value for {feature.replace('_', ' ')}.", fields=bundle["features"], form=request.form)
        if feature in {"year", "km_driven", "owner", "present_price"}:
            try:
                values[feature] = float(raw)
            except ValueError:
                return render_template("prediction.html", error=f"{feature.replace('_', ' ').title()} must be a number.", fields=bundle["features"], form=request.form)
            if values[feature] < 0:
                return render_template("prediction.html", error=f"{feature.replace('_', ' ').title()} cannot be negative.", fields=bundle["features"], form=request.form)
        else:
            values[feature] = raw
    try:
        prediction = float(bundle["pipeline"].predict(pd.DataFrame([values], columns=bundle["features"]))[0])
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("prediction.html", error="The submitted values could not be processed by the trained model.", fields=bundle["features"], form=request.form)
    return render_template("result.html", prediction=max(0, prediction), details=values)


@app.errorhandler(404)
def not_found(_):
    return render_template("base.html", content="Page not found."), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
