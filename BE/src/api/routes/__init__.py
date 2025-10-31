"""
API Routes initialization
"""
from .random_forest import router as random_forest_router
from .knn import router as knn_router

__all__ = ['random_forest_router', 'knn_router']
