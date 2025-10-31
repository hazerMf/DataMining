from functools import lru_cache
from models.random_forest_model import RandomForestModel
from models.knn_model import KNNSystolicModel, KNNDiastolicModel
from config import Config


# Singleton instances
_random_forest_model = None
_knn_systolic_model = None
_knn_diastolic_model = None


@lru_cache()
def get_random_forest_model() -> RandomForestModel:
    """
    Lấy singleton instance của Random Forest Model
    
    Returns:
        RandomForestModel instance
    """
    global _random_forest_model
    if _random_forest_model is None:
        _random_forest_model = RandomForestModel(
            model_path=Config.RANDOM_FOREST_MODEL_PATH,
            scaler_path=Config.RANDOM_FOREST_SCALER_PATH
        )
    return _random_forest_model


@lru_cache()
def get_knn_systolic_model() -> KNNSystolicModel:
    """
    Get singleton instance of KNN Systolic BP Model
    
    Returns:
        KNNSystolicModel instance
    """
    global _knn_systolic_model
    if _knn_systolic_model is None:
        _knn_systolic_model = KNNSystolicModel(
            model_path=Config.KNN_SYSTOLIC_MODEL_PATH,
            scaler_path=Config.KNN_SYSTOLIC_SCALER_PATH
        )
    return _knn_systolic_model


@lru_cache()
def get_knn_diastolic_model() -> KNNDiastolicModel:
    """
    Get singleton instance of KNN Diastolic BP Model
    
    Returns:
        KNNDiastolicModel instance
    """
    global _knn_diastolic_model
    if _knn_diastolic_model is None:
        _knn_diastolic_model = KNNDiastolicModel(
            model_path=Config.KNN_DIASTOLIC_MODEL_PATH,
            scaler_path=Config.KNN_DIASTOLIC_SCALER_PATH
        )
    return _knn_diastolic_model


def reload_models():
    """Reload all models"""
    global _random_forest_model, _knn_systolic_model, _knn_diastolic_model
    _random_forest_model = None
    _knn_systolic_model = None
    _knn_diastolic_model = None
    get_random_forest_model.cache_clear()
    get_knn_systolic_model.cache_clear()
    get_knn_diastolic_model.cache_clear()
