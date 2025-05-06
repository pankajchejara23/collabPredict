from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from src.utils.config import settings
from typing import Dict
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Pydantic Models ---
class PredictionInput(BaseModel):
    user_add_mean: float
    user_del_mean: float
    user_speak_mean: float
    user_turns_mean: float
    user_wh_mean: float
    user_self_mean: float
    user_us_mean: float
    user_add_sd: float
    user_del_sd: float
    user_speak_sd: float
    user_turns_sd: float
    user_wh_sd: float
    user_self_sd: float
    user_us_sd: float

class PredictionOutput(BaseModel):
    collaboration_quality: Dict[str, float]
    argumentation: Dict[str, float]
    mutual_understanding: Dict[str, float]
    problem_solving: Dict[str, float]

# --- Load Models ---
MODELS = {
    "collaboration_quality": joblib.load(settings.paths.MODEL_CQ),
    "argumentation": joblib.load(settings.paths.MODEL_ARG),
    "mutual_understanding": joblib.load(settings.paths.MODEL_SMU),
    "problem_solving": joblib.load(settings.paths.MODEL_STR)
}

@router.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make predictions across all collaboration metrics.
    Returns:
        {
            "collaboration_quality": {"prediction": 1, "probability": 0.85},
            "argumentation": {...},
            ...
        }
    """
    try:
        # Convert input to DataFrame
        features = pd.DataFrame([input_data.dict()])
        
        results = {}
        for target, model in MODELS.items():
            prediction = int(model.predict(features)[0])
            probability = float(model.predict_proba(features)[0, 1])
            results[target] = {
                "prediction": prediction,
                "probability": probability
            }
        
        logger.info(f"Prediction successful for input: {input_data.dict()}")
        return results

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Prediction error: {str(e)}"
        )