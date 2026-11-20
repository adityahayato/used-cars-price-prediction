"""Train and evaluate the Used Car Price Prediction models.

The script expects the genuine CarDekho CSV at data/car_data.csv. It never
creates substitute records: a clear error is raised when the dataset is absent
or its target column cannot be identified.
"""
from pathlib import Path
import argparse
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "car_data.csv"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "best_model.pkl"

ALIASES = {
    "car_name": ["car_name", "name", "car name"],
    "year": ["year", "manufacturing_year", "manufacture_year"],
    "selling_price": ["selling_price", "selling price", "price", "sellingprice"],
    "present_price": ["present_price", "present price", "presentprice"],
    "km_driven": ["km_driven", "km driven", "kilometers_driven", "kms_driven"],
    "fuel_type": ["fuel_type", "fuel type", "fuel"],
    "seller_type": ["seller_type", "seller type", "seller"],
    "transmission": ["transmission", "transmission_type", "transmission type"],
    "owner": ["owner", "owners", "previous_owners", "previous owners"],
}


def normalise(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(columns, candidates):
    lookup = {normalise(column): column for column in columns}
    for candidate in candidates:
        if normalise(candidate) in lookup:
            return lookup[normalise(candidate)]
    return None


def load_and_prepare(path=DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the genuine CarDekho CSV and "
            "save it as data/car_data.csv before training."
        )
    frame = pd.read_csv(path)
    mapping = {key: find_column(frame.columns, values) for key, values in ALIASES.items()}
    if not mapping["selling_price"]:
        raise ValueError(
            "Could not find a selling price column. Expected one of: "
            + ", ".join(ALIASES["selling_price"])
        )
    frame = frame.rename(columns={source: key for key, source in mapping.items() if source})
    selected = [key for key in ALIASES if key in frame.columns]
    frame = frame[selected].drop_duplicates().copy()
    for column in ["year", "selling_price", "present_price", "km_driven", "owner"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "selling_price" not in frame:
        raise ValueError("The dataset must contain selling_price after column adaptation.")
    frame = frame[frame["selling_price"] > 0]
    for column in ["year", "km_driven", "owner", "present_price"]:
        if column in frame:
            frame = frame[(frame[column].isna()) | (frame[column] >= 0)]
    frame = frame.dropna(subset=["selling_price"])
    if len(frame) < 20:
        raise ValueError("The validated dataset has fewer than 20 usable rows.")
    return frame, mapping


def make_eda(frame):
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    plots = []
    plt.figure(figsize=(8, 5)); sns.histplot(frame["selling_price"], kde=True, color="#21c7a8"); plt.title("Selling Price Distribution"); plots.append("price_distribution.png")
    plt.figure(figsize=(8, 5)); sns.countplot(data=frame, x="year", color="#3686ff"); plt.xticks(rotation=45); plt.title("Cars by Manufacturing Year"); plots.append("manufacturing_year.png")
    if "fuel_type" in frame:
        plt.figure(figsize=(7, 5)); sns.countplot(data=frame, x="fuel_type", color="#21c7a8"); plt.title("Fuel Type Distribution"); plots.append("fuel_type_distribution.png")
    if "transmission" in frame:
        plt.figure(figsize=(7, 5)); sns.countplot(data=frame, x="transmission", color="#3686ff"); plt.title("Transmission Distribution"); plots.append("transmission_distribution.png")
    if "km_driven" in frame:
        plt.figure(figsize=(8, 5)); sns.scatterplot(data=frame, x="km_driven", y="selling_price", alpha=.65, color="#21c7a8"); plt.title("Kilometers Driven vs Selling Price"); plots.append("km_vs_price.png")
    numeric = frame.select_dtypes(include=np.number)
    if numeric.shape[1] > 1:
        plt.figure(figsize=(9, 6)); sns.heatmap(numeric.corr(), annot=True, cmap="crest", fmt=".2f"); plt.title("Numerical Correlation Heatmap"); plots.append("correlation_heatmap.png")
    for filename in plots:
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / filename, dpi=130); plt.close()


def train(path=DATA_PATH):
    frame, mapping = load_and_prepare(path)
    make_eda(frame)
    X = frame.drop(columns="selling_price")
    y = frame["selling_price"]
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=12),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1, max_depth=20),
    }
    rows, fitted = [], {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        rows.append({"model": name, "r2_score": r2_score(y_test, prediction), "mae": mean_absolute_error(y_test, prediction), "rmse": mean_squared_error(y_test, prediction, squared=False)})
        fitted[name] = pipeline
    results = pd.DataFrame(rows).sort_values(["r2_score", "rmse"], ascending=[False, True])
    OUTPUT_DIR.mkdir(exist_ok=True); results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False); results.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False)
    best_name = results.iloc[0]["model"]
    joblib.dump({"pipeline": fitted[best_name], "features": X.columns.tolist(), "feature_defaults": {column: (frame[column].dropna().mode().iloc[0] if not frame[column].dropna().empty else "") for column in X.columns}, "model_name": best_name, "column_mapping": mapping}, MODEL_PATH)
    print(f"Trained {len(frame):,} rows. Best model: {best_name}")
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train used-car price models.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the genuine CSV dataset")
    args = parser.parse_args()
    train(args.data)
