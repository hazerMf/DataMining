from functools import lru_cache
from models.random_forest_model import RandomForestModel
from models.model_2 import Model2
from config import Config


# Singleton instances
_random_forest_model = None
_model_2 = None


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
def get_model_2() -> Model2:
    """
    Lấy singleton instance của Model 2
    
    Returns:
        Model2 instance
    """
    global _model_2
    if _model_2 is None:
        # TODO: Cập nhật path khi có model thực tế
        _model_2 = Model2(
            model_path=Config.MODEL_2_PATH
        )
    return _model_2


def reload_models():
    """Reload tất cả models"""
    global _random_forest_model, _model_2
    _random_forest_model = None
    _model_2 = None
    get_random_forest_model.cache_clear()
    get_model_2.cache_clear()
