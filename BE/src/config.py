import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
  
    # API Settings
    API_V1_STR = "/api/v1"
    PROJECT_NAME = "Multi-Model ML Prediction Service"
    VERSION = "2.0.0"
    DESCRIPTION = "FastAPI backend hỗ trợ nhiều ML models cho các bài toán dự đoán khác nhau"
    
    # Random Forest Model Paths
    RANDOM_FOREST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest', 'random_forest.joblib')
    RANDOM_FOREST_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'random_forest', 'scaler.pkl')
    
    # Model 2 Paths (Cập nhật khi có model thực tế)
    MODEL_2_PATH = os.path.join(BASE_DIR, 'models', 'model_2', 'model.joblib')
    
    # Server Settings
    HOST = "0.0.0.0"
    PORT = 8000
    RELOAD = True  
    
    # CORS Settings
    ALLOWED_ORIGINS = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Logging
    LOG_LEVEL = "INFO"