import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", "/app/models/model.joblib"))
    app.state.bundle = joblib.load(path)
    yield


app = FastAPI(title="Iris Prediction API", version="1.0.0", lifespan=lifespan)


class Measurements(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sepal_length: float = Field(gt=0, le=30, examples=[5.1])
    sepal_width: float = Field(gt=0, le=30, examples=[3.5])
    petal_length: float = Field(gt=0, le=30, examples=[1.4])
    petal_width: float = Field(gt=0, le=30, examples=[0.2])


class Prediction(BaseModel):
    species: str
    probabilities: dict[str, float]
    run_id: str


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "run_id": request.app.state.bundle["metadata"]["run_id"]}


@app.get("/model")
def model_info(request: Request):
    return request.app.state.bundle["metadata"]


@app.post("/predict", response_model=Prediction)
def predict(values: Measurements, request: Request):
    bundle = request.app.state.bundle
    features = bundle["metadata"]["features"]
    frame = pd.DataFrame([values.model_dump()], columns=features)
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(frame)[0]
    return Prediction(
        species=str(pipeline.predict(frame)[0]),
        probabilities=dict(zip(pipeline.classes_.tolist(), probabilities.tolist())),
        run_id=bundle["metadata"]["run_id"],
    )
