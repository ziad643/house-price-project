"""Create a small development-only model so the API/UI can be exercised before Kaggle data is available.

Run notebooks/train_house_price_model.py after downloading the required dataset; that replaces this artifact.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

root = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(42)
locations = ["Andheri", "Bandra", "Dwarka", "Indiranagar", "Koramangala", "other"]
n = 360
frame = pd.DataFrame({
    "carpet_area_sqft": rng.integers(350, 2400, n), "floor_num": rng.integers(0, 20, n),
    "bathroom": rng.integers(1, 5, n), "balcony": rng.integers(0, 3, n),
    "location_grouped": rng.choice(locations, n), "Furnishing": rng.choice(["Furnished", "Semi-Furnished", "Unfurnished"], n),
    "Transaction": rng.choice(["New Property", "Resale"], n), "Ownership": rng.choice(["Freehold", "Leasehold", "Unknown"], n),
    "facing": rng.choice(["East", "West", "North", "South", "Unknown"], n),
})
location_bonus = frame.location_grouped.map({"Bandra": 7_000_000, "Andheri": 3_200_000, "Koramangala": 3_800_000, "Indiranagar": 3_400_000, "Dwarka": 1_500_000, "other": 1_200_000})
y = frame.carpet_area_sqft * 5_400 + frame.bathroom * 280_000 + location_bonus + rng.normal(0, 550_000, n)
numeric = ["carpet_area_sqft", "floor_num", "bathroom", "balcony"]
categorical = ["location_grouped", "Furnishing", "Transaction", "Ownership", "facing"]
prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), numeric), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)])
model = Pipeline([("prep", prep), ("reg", RandomForestRegressor(n_estimators=100, random_state=42))]).fit(frame, y)
models = root / "models"; models.mkdir(exist_ok=True)
joblib.dump(model, models / "house_price.pkl")
(models / "locations.json").write_text(json.dumps(locations), encoding="utf-8")
