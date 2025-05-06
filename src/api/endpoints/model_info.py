from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import json
from typing import Dict, List
from src.utils import config
import logging

logger = logging.getLogger(__name__)
settings = config.settings

router = APIRouter()

# --- Pydantic Models ---
class ModelMetadata(BaseModel):
    model_type: str
    n_features: int
    feature_names: List[str]
    training_date: str
    metrics: Dict[str, float]  # e.g. {"accuracy": 0.85, "f1": 0.82}

class ModelInfoResponse(BaseModel):
    collaboration_quality: ModelMetadata
    argumentation: ModelMetadata
    mutual_understanding: ModelMetadata
    problem_solving: ModelMetadata
    api_version: str

# --- Load Model Metadata ---
def load_model_metadata(model_path: str) -> Dict:
    """Helper to load model metadata"""
    try:
        model = joblib.load(model_path)
        return {
            "model_type": model.__class__.__name__,
            "n_features": len(model.feature_names_in_),
            "feature_names": list(model.feature_names_in_),
            "training_date": "2023-11-15",  # Should be saved during training
            "metrics": model.metrics_ if hasattr(model, "metrics_") else {}
        }
    except Exception as e:
        logger.error(f"Failed to load model metadata: {str(e)}")
        raise HTTPException(status_code=500, detail="Model metadata unavailable")

@router.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Returns metadata for all deployed models including:
    - Model type
    - Number of features
    - Feature names
    - Training metrics
    - API version
    """
    try:
        return {
            "collaboration_quality": load_model_metadata(settings.paths.MODEL_CQ),
            "argumentation": load_model_metadata(settings.paths.MODEL_ARG),
            "mutual_understanding": load_model_metadata(settings.paths.MODEL_SMU),
            "problem_solving": load_model_metadata(settings.paths.MODEL_STR),
            "api_version": settings.api.API_VERSION
        }
    except Exception as e:
        logger.error(f"Model info endpoint failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve model information"
        )