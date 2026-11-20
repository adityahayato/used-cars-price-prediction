from pathlib import Path
from datetime import datetime
import logging
import joblib
import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"
app = Flask(__name__)
app.config["SECRET_KEY"] = "used-car-prediction-local"
logging.basicConfig(level=logging.INFO)

CAR_LAUNCH_YEARS = {
    "maruti swift dzire vdi": 2008,
    "maruti swift": 2005,
    "maruti alto 800": 2012,
    "maruti wagon r": 1999,
    "hyundai i20 sportz": 2008,
    "hyundai creta": 2015,
    "hyundai grand i10": 2013,
    "honda city i vtec": 1998,
    "honda amaze": 2013,
    "toyota innova 2.5 g (diesel) 7 seater": 2005,
    "toyota fortuner": 2009,
    "ford ecosport 1.5 diesel titanium": 2013,
    "ford figo": 2010,
    "renault kwid rxt": 2015,
    "renault duster": 2012,
    "volkswagen vento 1.5 tdi highline": 2010,
    "volkswagen polo": 2010,
    "tata nexon": 2017,
    "tata tiago": 2016,
    "tata tigor": 2017,
    "mahindra scorpio": 2002,
    "mahindra xuv500": 2011,
    "mahindra thar": 2010,
    "force gurkha": 1998,
    "datsun go": 2014,
    "isuzu d-max": 2012,
    "skoda kushaq": 2021,
    "mg hector": 2019,
    "kia seltos": 2019,
    "byd atto 3": 2022,
    "tesla model 3": 2017,
    "toyota camry hybrid": 2006,
}

CAR_SPECS = {
    "maruti swift dzire vdi": {"fuel": {"Diesel"}, "transmission": {"Manual"}},
    "maruti swift": {"fuel": {"Petrol", "Diesel", "CNG"}, "transmission": {"Manual", "Automatic"}},
    "maruti alto 800": {"fuel": {"Petrol", "CNG"}, "transmission": {"Manual"}},
    "maruti wagon r": {"fuel": {"Petrol", "CNG"}, "transmission": {"Manual", "Automatic"}},
    "hyundai i20 sportz": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "hyundai creta": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "hyundai grand i10": {"fuel": {"Petrol", "CNG"}, "transmission": {"Manual", "Automatic"}},
    "honda city i vtec": {"fuel": {"Petrol"}, "transmission": {"Manual", "Automatic"}},
    "honda amaze": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "toyota innova 2.5 g (diesel) 7 seater": {"fuel": {"Diesel"}, "transmission": {"Manual"}},
    "toyota fortuner": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "toyota camry hybrid": {"fuel": {"Petrol", "Electric"}, "transmission": {"Automatic"}},
    "ford ecosport 1.5 diesel titanium": {"fuel": {"Diesel"}, "transmission": {"Manual"}},
    "ford figo": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "renault kwid rxt": {"fuel": {"Petrol"}, "transmission": {"Manual", "Automatic"}},
    "renault duster": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "volkswagen vento 1.5 tdi highline": {"fuel": {"Diesel"}, "transmission": {"Manual", "Automatic"}},
    "volkswagen polo": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "tata nexon": {"fuel": {"Petrol", "Diesel", "Electric"}, "transmission": {"Manual", "Automatic"}},
    "tata tiago": {"fuel": {"Petrol", "CNG", "Electric"}, "transmission": {"Manual", "Automatic"}},
    "tata tigor": {"fuel": {"Petrol", "CNG", "Electric"}, "transmission": {"Manual", "Automatic"}},
    "mahindra scorpio": {"fuel": {"Diesel"}, "transmission": {"Manual"}},
    "mahindra xuv500": {"fuel": {"Diesel"}, "transmission": {"Manual", "Automatic"}},
    "mahindra thar": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "force gurkha": {"fuel": {"Diesel"}, "transmission": {"Manual"}},
    "datsun go": {"fuel": {"Petrol"}, "transmission": {"Manual"}},
    "isuzu d-max": {"fuel": {"Diesel"}, "transmission": {"Manual", "Automatic"}},
    "skoda kushaq": {"fuel": {"Petrol"}, "transmission": {"Manual", "Automatic"}},
    "mg hector": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "kia seltos": {"fuel": {"Petrol", "Diesel"}, "transmission": {"Manual", "Automatic"}},
    "byd atto 3": {"fuel": {"Electric"}, "transmission": {"Automatic"}},
    "tesla model 3": {"fuel": {"Electric"}, "transmission": {"Automatic"}},
}

# First model years in the India market where an automatic/AMT/DCT variant
# was available for the suggested model. Earlier years remain manual-only.
AUTOMATIC_GEARBOX_START_YEARS = {
    "maruti swift": 2007,
    "maruti wagon r": 2015,
    "hyundai i20 sportz": 2014,
    "hyundai creta": 2015,
    "hyundai grand i10": 2014,
    "honda city i vtec": 2014,
    "honda amaze": 2016,
    "toyota fortuner": 2009,
    "ford figo": 2016,
    "renault kwid rxt": 2016,
    "renault duster": 2016,
    "volkswagen vento 1.5 tdi highline": 2011,
    "volkswagen polo": 2014,
    "tata nexon": 2018,
    "tata tiago": 2017,
    "tata tigor": 2018,
    "mahindra xuv500": 2015,
    "mahindra thar": 2020,
    "isuzu d-max": 2016,
    "skoda kushaq": 2021,
    "mg hector": 2019,
    "kia seltos": 2019,
    "byd atto 3": 2022,
    "tesla model 3": 2017,
    "toyota camry hybrid": 2006,
}


def format_lakh(value):
    amount = str(int(round(float(value))))
    if len(amount) <= 3:
        return f"₹ {amount}"
    last_three = amount[-3:]
    remaining = amount[:-3]
    groups = []
    while remaining:
        groups.insert(0, remaining[-2:])
        remaining = remaining[:-2]
    return f"₹ {','.join(groups)},{last_three}"


@app.context_processor
def form_options():
    return {
        "current_year": datetime.now().year,
        "format_lakh": format_lakh,
        "choice_options": {
            "fuel_type": ["Petrol", "Diesel", "CNG", "LPG", "Electric"],
            "seller_type": ["Dealer", "Individual", "Trustmark Dealer"],
            "transmission": ["Manual", "Automatic"],
            "owner": ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"],
        },
        "car_name_suggestions": [
            "Maruti Swift Dzire VDI", "Maruti Swift", "Maruti Alto 800", "Maruti Wagon R",
            "Hyundai i20 Sportz", "Hyundai Creta", "Hyundai Grand i10", "Honda City i VTEC", "Honda Amaze",
            "Toyota Innova 2.5 G (Diesel) 7 Seater", "Toyota Fortuner", "Toyota Camry Hybrid",
            "Ford Ecosport 1.5 Diesel Titanium", "Ford Figo", "Renault KWID RXT", "Renault Duster",
            "Volkswagen Vento 1.5 TDI Highline", "Volkswagen Polo", "Tata Nexon", "Tata Tiago", "Tata Tigor",
            "Mahindra Scorpio", "Mahindra XUV500", "Mahindra Thar", "Force Gurkha", "Datsun GO",
            "Isuzu D-Max", "Skoda Kushaq", "MG Hector", "Kia Seltos", "BYD Atto 3", "Tesla Model 3",
        ],
        "car_launch_years": CAR_LAUNCH_YEARS,
    }


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
        return render_template("prediction.html", fields=bundle["features"], numeric_features=bundle.get("numeric_features", []), defaults=bundle.get("feature_defaults", {}))
    values = {}
    numeric_features = set(bundle.get("numeric_features", []))
    for feature in bundle["features"]:
        raw = request.form.get(feature, "").strip()
        if raw == "":
            return render_template("prediction.html", error=f"Please provide a value for {feature.replace('_', ' ')}.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
        if feature in numeric_features:
            try:
                values[feature] = float(raw)
            except ValueError:
                return render_template("prediction.html", error=f"{feature.replace('_', ' ').title()} must be a number.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
            if values[feature] < 0:
                return render_template("prediction.html", error=f"{feature.replace('_', ' ').title()} cannot be negative.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
            if feature == "year":
                if not values[feature].is_integer():
                    return render_template("prediction.html", error="Year must be a whole number.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
                if values[feature] < 1950 or values[feature] > datetime.now().year:
                    return render_template("prediction.html", error=f"Year must be between 1950 and {datetime.now().year}.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
        else:
            values[feature] = raw
    launch_year = CAR_LAUNCH_YEARS.get(str(values.get("car_name", "")).strip().lower())
    if launch_year and values.get("year", 0) < launch_year:
        return render_template("prediction.html", error=f"{values['car_name']} was launched around {launch_year}. Please enter {launch_year} or a later year.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
    car_specs = CAR_SPECS.get(str(values.get("car_name", "")).strip().lower())
    if car_specs and values.get("fuel_type") not in car_specs["fuel"]:
        allowed = ", ".join(sorted(car_specs["fuel"]))
        return render_template("prediction.html", error=f"Fuel type mismatch: {values['car_name']} uses {allowed} in this model selection.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
    if car_specs and values.get("transmission") not in car_specs["transmission"]:
        allowed = ", ".join(sorted(car_specs["transmission"]))
        return render_template("prediction.html", error=f"Gearbox mismatch: {values['car_name']} is available with {allowed} in this model selection.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
    automatic_start_year = AUTOMATIC_GEARBOX_START_YEARS.get(str(values.get("car_name", "")).strip().lower())
    if values.get("transmission") == "Automatic" and automatic_start_year and values.get("year", 0) < automatic_start_year:
        return render_template("prediction.html", error=f"Automatic gearbox unavailable: {values['car_name']} automatic variants started around {automatic_start_year}. Choose Manual or enter {automatic_start_year} or later.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
    try:
        prediction = float(bundle["pipeline"].predict(pd.DataFrame([values], columns=bundle["features"]))[0])
    except Exception:
        app.logger.exception("Prediction failed")
        return render_template("prediction.html", error="The submitted values could not be processed by the trained model.", fields=bundle["features"], numeric_features=numeric_features, form=request.form)
    return render_template("result.html", prediction=max(0, prediction), details=values)


@app.errorhandler(404)
def not_found(_):
    return render_template("base.html", content="Page not found."), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
