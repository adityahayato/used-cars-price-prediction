# Used Car Price Prediction System

AutoValue is a local Flask application that estimates a used car's resale value in Indian Rupees from its specifications. It is designed as a beginner-friendly internship project with a reproducible data-to-web workflow.

## Features

- Genuine CarDekho CSV input with schema detection and validation
- Missing-value handling, duplicate removal, numeric validation and categorical encoding
- EDA charts: price, year, fuel, transmission, kilometres and correlations
- Comparison of Linear Regression, Decision Tree and Random Forest
- R², MAE and RMSE evaluation exported to CSV
- Joblib-saved preprocessing-plus-model pipeline
- Responsive dark navy, mint and blue Flask UI
- No database, paid API, JavaScript framework or virtual environment required

## Technology

Python, Flask, Pandas, NumPy, scikit-learn, Joblib, Matplotlib, Seaborn, HTML5, CSS3 and vanilla JavaScript.

## Folder structure

```text
Used_Car_Price_Prediction/
├── app.py                 # Flask server and validation
├── train_model.py         # data, EDA, training and evaluation
├── data/car_data.csv      # genuine CarDekho dataset
├── models/best_model.pkl  # generated after training (not committed)
├── outputs/               # generated charts and metrics
├── notebooks/eda.ipynb
├── templates/             # Flask pages
├── static/                # CSS and JavaScript
├── install.bat
└── run.bat
```

## Dataset setup

The project uses the genuine `CAR DETAILS FROM CAR DEKHO.csv` file from [Kaggle: Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho), copied to `data/car_data.csv`. It contains 4,340 records and the columns `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, and `owner`.

The trainer detects common names such as `Car_Name`/`name`, `Year`, `Selling_Price`, `Present_Price`, `Kms_Driven`, `Fuel_Type`/`fuel`, `Seller_Type`, `Transmission` and `Owner`. It does not fabricate records. If the target column is absent, it stops with an actionable error.

## Install and train on Windows

1. Install Python 3.10 or newer from python.org and enable “Add Python to PATH”. This project intentionally uses the normal Python installation; do not create a virtual environment.
2. Open the project folder and double-click `install.bat`, or run `python -m pip install -r requirements.txt`.
3. The genuine dataset is already at `data\car_data.csv` in this project. If it is missing in a fresh copy, download it from the Kaggle link above.
4. Run `python train_model.py`.
5. Check `outputs\model_comparison.csv`, `outputs\evaluation_results.csv` and `outputs\eda\`. The best pipeline is saved to `models\best_model.pkl`.

## Run the application

Run `run.bat`, then open http://127.0.0.1:5000. Alternatively run `python app.py`. Debug mode is disabled by default. If the model is missing, the prediction page explains how to train it.

## Evaluation

The data is split reproducibly with `random_state=42`. R² measures explained variance (higher is better); MAE is average absolute error in dataset price units; RMSE penalizes large errors more strongly. The script selects the highest R², using lower RMSE as a tie-breaker. Metrics are measured on the held-out test set and are never hardcoded or presented as a guarantee.

## How to use

Open “Estimate price”, enter the fields shown by the trained dataset, submit, and review the estimated resale value plus the submitted details. Treat the result as a data-informed estimate; inspection, location, negotiation and market demand affect actual price.

## Common errors

- **Dataset not found:** save the genuine CSV exactly as `data/car_data.csv`.
- **Selling price column missing:** use the CarDekho dataset or rename its target column to `Selling_Price`.
- **Model missing:** install packages, place the CSV, then run `python train_model.py`.
- **Module not found:** run `install.bat` from this folder with internet access.
- **Port already in use:** close the other local Flask process, or change the port in the last line of `app.py`.
