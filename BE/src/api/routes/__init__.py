"""
API Routes initialization
"""
from .random_forest import router as random_forest_router
from .model_2 import router as model_2_router

__all__ = ['random_forest_router', 'model_2_router']
