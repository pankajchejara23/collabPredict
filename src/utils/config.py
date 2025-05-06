import os
from pathlib import Path
from typing import Dict, Any
from pydantic import Field

from pydantic_settings import BaseSettings

# --- Project Structure Config ---
class PathConfig:
    """Paths to directories and files"""
    BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Project root
    
    # Data paths
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA = DATA_DIR / "raw/team_metrics.csv"
    PROCESSED_DATA = DATA_DIR / "processed/data.csv"
    FEATURE_NAMES = DATA_DIR / "processed/feature_columns.json"
    
    # Model paths
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = MODELS_DIR / "model_evaluation"
    MODEL_CQ = MODELS_DIR / "trained_models/random_forest_CQ.joblib"
    MODEL_ARG = MODELS_DIR / "trained_models/random_forest_ARG.joblib"
    MODEL_SMU = MODELS_DIR / "trained_models/random_forest_SMU.joblib"
    MODEL_STR = MODELS_DIR / "trained_models/random_forest_STR.joblib"

# --- API Config ---
class APIConfig(BaseSettings):
    """FastAPI settings (can load from environment variables)"""
    API_TITLE: str = "Collaboration Quality Prediction API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8090
    DEBUG_MODE: bool = False

    class Config:
        env_prefix = "API_"  # Environment variables will override defaults if set (e.g., `API_PORT=9000`)

# --- Model Training Config ---
class ModelConfig:
    """Hyperparameters and training settings"""
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # RandomForest defaults (override in training scripts)
    RF_PARAMS: Dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 5,
        "class_weight": "balanced"
    }

# --- Logging Config ---
class LoggingConfig:
    """Logging settings"""
    LOG_DIR = PathConfig.BASE_DIR / "logs"
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- Combine all configs for easy access ---
class Settings:
    paths = PathConfig()
    api = APIConfig()
    model = ModelConfig()
    logging = LoggingConfig()

# Singleton instance to import elsewhere
settings = Settings()