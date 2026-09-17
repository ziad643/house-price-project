import pandas as pd

from app.schemas.prediction import PredictionRequest


def request_to_frame(request: PredictionRequest, allowed_locations: set[str]) -> pd.DataFrame:
    """Keep this order in sync with the training pipeline."""
    location = request.location if request.location in allowed_locations else "other"
    return pd.DataFrame([{
        "carpet_area_sqft": request.carpet_area_sqft,
        "floor_num": request.floor_num,
        "bathroom": request.bathroom,
        "balcony": request.balcony,
        "location_grouped": location,
        "Furnishing": request.furnishing,
        "Transaction": request.transaction,
        "Ownership": request.ownership,
        "facing": request.facing,
    }])

