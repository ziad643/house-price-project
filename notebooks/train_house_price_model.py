"""Train the regression pipeline used by the API.

Put Kaggle's house_prices.csv in notebooks/data/ before running this script.
The notebook follows the same steps with EDA and model comparison.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "notebooks" / "data" / "house_prices.csv"
MODEL_DIR = ROOT / "backend" / "models"
NUMERIC = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
CATEGORICAL = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]


def parse_price(value):
    if not isinstance(value, str): return np.nan
    value = value.lower().replace(",", "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(lac|lakh|cr|crore)?", value)
    if not match: return np.nan
    number, unit = match.groups()
    return float(number) * {"lac": 1e5, "lakh": 1e5, "cr": 1e7, "crore": 1e7}.get(unit, 1)


def parse_area(value):
    if not isinstance(value, str): return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)\s*(sq\.?\s*m|sqm|sq\.?\s*ft|sqft)?", value.lower().replace(",", ""))
    if not match: return np.nan
    value, unit = match.groups()
    return float(value) * 10.764 if unit and ("m" in unit and "ft" not in unit) else float(value)


def parse_number(value):
    if isinstance(value, (int, float)): return value
    match = re.search(r"\d+", str(value))
    return float(match.group()) if match else np.nan


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["price_clean"] = df["Amount(in rupees)"].map(parse_price)
    df["carpet_area_sqft"] = df.get("Carpet Area", pd.Series(index=df.index)).map(parse_area)
    df["floor_num"] = df.get("Floor", pd.Series(index=df.index)).map(parse_number)
    for column, target in [("Bathroom", "bathroom"), ("Balcony", "balcony")]:
        df[target] = df.get(column, pd.Series(index=df.index)).map(parse_number)
    df = df.dropna(subset=["price_clean", "carpet_area_sqft"])
    df["location_grouped"] = df["location"].fillna("other")
    popular = df["location_grouped"].value_counts().head(50).index
    df.loc[~df["location_grouped"].isin(popular), "location_grouped"] = "other"
    for column in ["Furnishing", "Transaction", "Ownership", "facing"]:
        df[column] = df.get(column, "Unknown").fillna("Unknown").astype(str)
    unit_price = df["price_clean"] / df["carpet_area_sqft"]
    df = df[unit_price.between(unit_price.quantile(.01), unit_price.quantile(.99))]
    return df


def make_pipeline(regressor):
    prep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", prep), ("reg", regressor)])


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset missing: {DATA_PATH}. Download it from Kaggle first.")
    df = clean(pd.read_csv(DATA_PATH))
    X, y = df[NUMERIC + CATEGORICAL], df["price_clean"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    candidates = {"ridge": Ridge(alpha=4), "random_forest": RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1)}
    fitted = {}
    for name, estimator in candidates.items():
        model = make_pipeline(estimator).fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(name, {"MAE": round(mean_absolute_error(y_test, prediction), 2),
                     "RMSE": round(mean_squared_error(y_test, prediction) ** .5, 2),
                     "R2": round(r2_score(y_test, prediction), 4)})
        fitted[name] = model
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(fitted["random_forest"], MODEL_DIR / "house_price.pkl")
    (MODEL_DIR / "locations.json").write_text(json.dumps(sorted(df.location_grouped.unique())), encoding="utf-8")


if __name__ == "__main__":
    main()
