from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    location: str = Field(min_length=1, max_length=120)
    carpet_area_sqft: float = Field(gt=0, le=100_000)
    floor_num: int = Field(ge=-5, le=200)
    bathroom: int = Field(ge=0, le=30)
    balcony: int = Field(ge=0, le=30)
    car_parking: int = Field(ge=0, le=30)
    furnishing: str = Field(min_length=1)
    transaction: str = Field(min_length=1)
    ownership: str = Field(min_length=1)
    facing: str = Field(min_length=1)


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "INR"
