import json
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(
    MODEL_DIR / "wine_random_forest_model.joblib"
)

with open(MODEL_DIR / "model_params.json") as f:
    model_info = json.load(f)

app = FastAPI(title="Wine Classification API")


class WineInput(BaseModel):
    alcohol: float = Field(gt=0, le=30)
    color_intensity: float = Field(gt=0, le=30)
    proline: float = Field(gt=0, le=2000)


@app.get("/")
def home():
    return {"message": "Wine Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: WineInput):
    features = pd.DataFrame(
        [[data.alcohol, data.color_intensity, data.proline]],
        columns=model_info["features"]
    )

    prediction = int(model.predict(features)[0])

    return {
        "prediction": prediction,
        "class_name": f"Wine Class {prediction}"
    }