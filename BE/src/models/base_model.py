from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pandas as pd


class BaseModel(ABC):
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize base model
        
        Args:
            model_path: Đường dẫn đến file model
            **kwargs: Các tham số khác (scaler_path, etc.)
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """Load model và các components cần thiết"""
        pass
    
    @abstractmethod
    def predict(self, input_data: pd.DataFrame) -> Tuple[Any, float]:
        """
        Thực hiện prediction
        
        Args:
            input_data: DataFrame chứa dữ liệu đầu vào
            
        Returns:
            Tuple[prediction, probability]
        """
        pass
    
    @abstractmethod
    def preprocess(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu trước khi predict
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed data
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Trả về thông tin về model"""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "model_type": self.__class__.__name__
        }
