"""
Random Forest Model - Model dự đoán tăng huyết áp
"""
import os
from typing import Tuple, Any
import pandas as pd
from joblib import load
from .base_model import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, model_path: str, scaler_path: str = None):
        """
        Initialize Random Forest Model
        
        Args:
            model_path: Đường dẫn đến file random_forest.joblib
            scaler_path: Đường dẫn đến file scaler.pkl (optional)
        """
        super().__init__(model_path)
        self.scaler_path = scaler_path
        self.scaler = None
        
        # Features cần scale (không bao gồm Sex)
        self.features_to_scale = ['Age', 'Height', 'Weight', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'BMI']
        
        # Thứ tự features đầy đủ (Sex + features_to_scale + categorical)
        self.feature_order = [
            'Sex', 'Age', 'Height', 'Weight', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'BMI',
            'Diabetes_Diabetes', 'Diabetes_None', 'Diabetes_Type2',
            'Cerebral_infarction_None', 'Cerebral_infarction_infarction',
            'Cerebrovascular_None', 'Cerebrovascular_disease', 'Cerebrovascular_insuff'
        ]
        
        self.label_map = {
            0: "Bình thường (Normal)",
            1: "Tiền tăng huyết áp (Prehypertension)",
            2: "Tăng huyết áp giai đoạn 1 (Stage 1 Hypertension)",
            3: "Tăng huyết áp giai đoạn 2 (Stage 2 Hypertension)"
        }
        self.load_model()
    
    def load_model(self) -> None:
        """Load Random Forest model và scaler"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            self.model = load(self.model_path)
            self.is_loaded = True
            print(f"Loaded Random Forest model from {self.model_path}")
            
            # Load scaler 
            if self.scaler_path and os.path.exists(self.scaler_path):
                self.scaler = load(self.scaler_path)
                print(f"Loaded scaler from {self.scaler_path}")
                print(f"ℹAPI can accept RAW input")
            else:
                print(f"No scaler found - API only accepts NORMALIZED input")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess(self, input_data: pd.DataFrame, is_raw: bool = False) -> pd.DataFrame:
        """
        Tiền xử lý dữ liệu
        
        Args:
            input_data: DataFrame với input features
            is_raw: True nếu input là RAW data (chưa normalized), False nếu đã normalized
            
        Returns:
            DataFrame đã được chuẩn hóa (normalized)
        
        Raises:
            ValueError: Nếu is_raw=True nhưng không có scaler
        """
        df = input_data.copy()
        
        if is_raw:
            # Input là RAW data, cần normalize
            if self.scaler is None:
                raise ValueError(
                    "Cannot process raw input: scaler not found. "
                    "Please provide normalized input or ensure scaler.pkl exists."
                )
            
            # Transform ONLY numeric features (excluding Sex)
            df[self.features_to_scale] = self.scaler.transform(df[self.features_to_scale])
            print(f"✅ Normalized raw input using scaler")
        
        # Rename columns to match training data format
        column_mapping = {
            'Diabetes_Type2': 'Diabetes_Type 2 Diabetes',
            'Cerebral_infarction_infarction': 'Cerebral_infarction_cerebral infarction',
            'Cerebrovascular_None': 'Cerebrovascular_disease_None',
            'Cerebrovascular_disease': 'Cerebrovascular_disease_cerebrovascular disease',
            'Cerebrovascular_insuff': 'Cerebrovascular_disease_insufficiency of cerebral blood supply'
        }
        df = df.rename(columns=column_mapping)
        
        # Update feature order with correct column names
        correct_feature_order = [
            'Sex', 'Age', 'Height', 'Weight', 'Systolic_BP', 'Diastolic_BP', 'Heart_Rate', 'BMI',
            'Diabetes_Diabetes', 'Diabetes_None', 'Diabetes_Type 2 Diabetes',
            'Cerebral_infarction_None', 'Cerebral_infarction_cerebral infarction',
            'Cerebrovascular_disease_None', 'Cerebrovascular_disease_cerebrovascular disease',
            'Cerebrovascular_disease_insufficiency of cerebral blood supply'
        ]
        
        # Đảm bảo thứ tự features đúng
        df = df[correct_feature_order]
        
        return df
    
    def predict(self, input_data: pd.DataFrame, is_raw: bool = False) -> Tuple[int, float]:
        """
        Dự đoán tăng huyết áp
        
        Args:
            input_data: DataFrame với các features
            is_raw: True nếu input là RAW data, False nếu đã normalized
            
        Returns:
            Tuple[prediction_class, probability]
        """
        if not self.is_loaded:
            raise RuntimeError("Model chưa được load. Gọi load_model() trước.")
        
        # Tiền xử lý dữ liệu
        processed_data = self.preprocess(input_data, is_raw=is_raw)
        
        # Dự đoán
        prediction = int(self.model.predict(processed_data)[0])
        
        # Lấy xác suất
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(processed_data)[0]
            probability = float(probabilities[prediction])
        else:
            probability = 0.0
        
        return prediction, probability
    
    def get_prediction_label(self, prediction: int) -> str:
        """
        Chuyển đổi prediction thành label có ý nghĩa
        
        Args:
            prediction: Prediction class (0-3)
            
        Returns:
            Label string
        """
        return self.label_map.get(prediction, "Không xác định")
