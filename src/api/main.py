from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.utils.config import settings  # Centralized config
from src.api.endpoints import predict, model_info  # Modular endpoints
import logging
import joblib
from typing import Dict, List

# Initialize logging
logging.basicConfig(
    level=settings.logging.LOG_LEVEL,
    format=settings.logging.LOG_FORMAT
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api.API_TITLE,
    version=settings.api.API_VERSION,
    docs_url="/docs",
    redoc_url=None
)

# Include endpoint routers
app.include_router(
    predict.router,
    prefix="/api/v1",
    tags=["Predictions"]
)
app.include_router(
    model_info.router,
    prefix="/api/v1",
    tags=["Model Metadata"]
)

@app.get("/health")
async def health_check():
    """Liveness probe endpoint"""
    return {"status": "healthy"}

## Adding model info endpoints
###############################
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
            "training_date": model.training_date_ if hasattr(model,'training_date_') else '',
            "metrics": model.metrics_ if hasattr(model, "metrics_") else {}
        }
    except Exception as e:
        logger.error(f"Failed to load model metadata: {str(e)}")
        raise HTTPException(status_code=500, detail="Model metadata unavailable")

@app.get("/model_info", response_model=ModelInfoResponse)
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


## Adding model info endpoints
###############################

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

@app.post("/predict", response_model=PredictionOutput)
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

# Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8090