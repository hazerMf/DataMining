import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
  
    # API Settings
    API_V1_STR = "/api/v1"
    PROJECT_NAME = "Blood Pressure Prediction"
    VERSION = "2.0.0"
    DESCRIPTION = ""
    
    # Random Forest Model Paths
    RANDOM_FOREST_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest', 'random_forest.joblib')
    RANDOM_FOREST_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'random_forest', 'scaler.pkl')
    
    # KNN Systolic BP Model Paths
    KNN_SYSTOLIC_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn', 'knn_systolic_bp.joblib')
    KNN_SYSTOLIC_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'knn', 'scaler_systolic_bp.pkl')
    
    # KNN Diastolic BP Model Paths
    KNN_DIASTOLIC_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn', 'knn_diastolic_bp.joblib')
    KNN_DIASTOLIC_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'knn', 'scaler_diastolic_bp.pkl')
    
    # Blood Pressure Denormalization Constants (from raw dataset statistics)
    SYSTOLIC_BP_MEAN = 127.945205  # mmHg
    SYSTOLIC_BP_STD = 20.377779    # mmHg
    DIASTOLIC_BP_MEAN = 71.849315  # mmHg
    DIASTOLIC_BP_STD = 11.111203   # mmHg
    
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