from fastapi import APIRouter, HTTPException, Request

from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.services.preprocessing import request_to_frame

router = APIRouter(tags=["prediction"])


@router.get("/health")
def health(request: Request) -> dict[str, str]:
    return {"status": "ok" if getattr(request.app.state, "predictor", None) else "unavailable"}


@router.get("/locations")
def locations(request: Request) -> list[str]:
    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        raise HTTPException(status_code=503, detail="Model has not been loaded")
    return sorted(predictor.locations)


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor = getattr(request.app.state, "predictor", None)
    if not predictor:
        raise HTTPException(status_code=503, detail="Model has not been loaded")
    frame = request_to_frame(payload, predictor.locations)
    return PredictionResponse(predicted_price=predictor.predict(frame))

