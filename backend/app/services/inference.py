import json
from pathlib import Path

import joblib


class HousePricePredictor:
    def __init__(self, model_path: Path, locations_path: Path):
        self.model = joblib.load(model_path)
        self.locations = set(json.loads(locations_path.read_text(encoding="utf-8")))

    def predict(self, frame) -> float:
        return float(self.model.predict(frame)[0])

