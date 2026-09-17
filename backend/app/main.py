from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.prediction import router
from app.core.config import settings
from app.services.inference import HousePricePredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = HousePricePredictor(settings.model_path, settings.locations_path)
    yield


app = FastAPI(title="House Price Prediction API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

