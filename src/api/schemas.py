from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

# --- Enums for Categorical Values ---
class ModelType(str, Enum):
    RANDOM_FOREST = "RandomForestClassifier"
    GRADIENT_BOOSTING = "GradientBoostingClassifier"

class MetricType(str, Enum):
    ACCURACY = "accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"

# --- Input/Output Schemas ---
class PredictionInput(BaseModel):
    """Schema for prediction input data"""
    user_add_mean: float = Field(..., ge=0, le=1, description="Mean additions per turn")
    user_del_mean: float = Field(..., ge=0, le=1, description="Mean deletions per turn")
    user_speak_mean: float = Field(..., ge=0, le=1, description="Mean speaking time")
    user_turns_mean: float = Field(..., ge=0, le=1, description="Mean turns taken")
    user_wh_mean: float = Field(..., ge=0, le=1, description="Mean wh-questions asked")
    user_self_mean: float = Field(..., ge=0, le=1, description="Mean self-references")
    user_us_mean: float = Field(..., ge=0, le=1, description="Mean group references")
    
    # Standard deviations
    user_add_sd: float = Field(..., ge=0, description="Standard deviation of additions")
    user_del_sd: float = Field(..., ge=0, description="Standard deviation of deletions")
    user_speak_sd: float = Field(..., ge=0, description="Standard deviation of speaking time")
    user_turns_sd: float = Field(..., ge=0, description="Standard deviation of turns")
    user_wh_sd: float = Field(..., ge=0, description="Standard deviation of wh-questions")
    user_self_sd: float = Field(..., ge=0, description="Standard deviation of self-references")
    user_us_sd: float = Field(..., ge=0, description="Standard deviation of group references")

class PredictionResult(BaseModel):
    """Schema for a single model's prediction"""
    prediction: int = Field(..., description="Binary prediction (0 or 1)")
    probability: float = Field(..., ge=0, le=1, description="Confidence score")
    confidence_interval: Optional[tuple[float, float]] = Field(
        None, 
        description="95% confidence interval"
    )

class PredictionOutput(BaseModel):
    """Combined output for all models"""
    collaboration_quality: PredictionResult
    argumentation: PredictionResult
    mutual_understanding: PredictionResult
    problem_solving: PredictionResult
    timestamp: datetime = Field(default_factory=datetime.now)

# --- Model Metadata Schemas ---
class ModelMetrics(BaseModel):
    """Schema for model performance metrics"""
    accuracy: float = Field(..., ge=0, le=1)
    f1: float = Field(..., ge=0, le=1)
    precision: Optional[float] = Field(None, ge=0, le=1)
    recall: Optional[float] = Field(None, ge=0, le=1)

class ModelMetadata(BaseModel):
    """Schema for model metadata"""
    model_type: ModelType
    version: str = Field("1.0.0", regex=r"^\d+\.\d+\.\d+$")
    features: List[str]
    training_date: datetime
    metrics: ModelMetrics
    hyperparameters: Dict[str, Any]

class ModelInfoResponse(BaseModel):
    """API response for /model_info endpoint"""
    models: Dict[str, ModelMetadata]  # Keys: "collaboration_quality", etc.
    api_version: str

# --- Error Schemas ---
class HTTPError(BaseModel):
    """Standard error response"""
    detail: str
    error_type: str
    timestamp: datetime = Field(default_factory=datetime.now)