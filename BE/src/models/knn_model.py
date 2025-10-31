import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from config import Config


class KNNSystolicModel:
 
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize KNN Systolic BP model
        
        Args:
            model_path: Path to the trained KNN model (.joblib)
            scaler_path: Path to the scaler (.pkl) for raw input normalization
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        
        # Denormalization constants
        self.target_mean = Config.SYSTOLIC_BP_MEAN
        self.target_std = Config.SYSTOLIC_BP_STD
        
        # Expected features (15 total, includes Diastolic_BP)
        self.expected_features = [
            'Sex', 'Age', 'Height', 'Weight', 'Diastolic_BP', 'Heart_Rate', 'BMI',
            'Diabetes_Diabetes', 'Diabetes_None', 'Diabetes_Type 2 Diabetes',
            'Cerebral_infarction_None', 'Cerebral_infarction_cerebral infarction',
            'Cerebrovascular_disease_None', 'Cerebrovascular_disease_cerebrovascular disease',
            'Cerebrovascular_disease_insufficiency of cerebral blood supply'
        ]
        
        # Numeric features that need scaling (6 features including Diastolic_BP)
        self.numeric_features = ['Age', 'Height', 'Weight', 'Diastolic_BP', 'Heart_Rate', 'BMI']
        
    def preprocess(self, input_data: Dict[str, Any], is_raw: bool = False) -> pd.DataFrame:
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Dictionary with feature values
            is_raw: If True, normalize numeric features using scaler
            
        Returns:
            DataFrame ready for model prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Ensure all expected features exist
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training data
        df = df[self.expected_features]
        
        # Normalize numeric features if raw input
        if is_raw and self.scaler:
            # Extract numeric features
            numeric_data = df[self.numeric_features].values
            
            # Scale numeric features
            scaled_numeric = self.scaler.transform(numeric_data)
            
            # Replace with scaled values
            df[self.numeric_features] = scaled_numeric
        
        return df
    
    def predict(self, input_data: Dict[str, Any], is_raw: bool = False) -> Dict[str, float]:
        """
        Predict Systolic BP with denormalization
        
        Args:
            input_data: Dictionary with feature values
            is_raw: If True, input will be normalized before prediction
            
        Returns:
            Dictionary with:
                - predicted_normalized: Predicted value in normalized form
                - predicted_value_mmHg: Predicted value in mmHg (denormalized)
                - confidence_interval_lower: Lower bound of 95% confidence interval (mmHg)
                - confidence_interval_upper: Upper bound of 95% confidence interval (mmHg)
                - prediction_std_normalized: Std of prediction (normalized)
                - prediction_std_mmHg: Std of prediction (mmHg)
        """
        # Preprocess input
        processed_data = self.preprocess(input_data, is_raw)
        
        # Get prediction (normalized)
        prediction_normalized = self.model.predict(processed_data)[0]
        
        # Denormalize prediction to mmHg
        prediction_mmHg = prediction_normalized * self.target_std + self.target_mean
        
        # Calculate prediction uncertainty using k nearest neighbors
        try:
            # Get k nearest neighbors and their target values
            distances, indices = self.model.kneighbors(processed_data)
            
            # Get the target values of k nearest neighbors
            # Note: self.model._y contains training targets
            neighbor_values = self.model._y[indices[0]]
            
            # Calculate standard deviation of neighbor predictions (normalized)
            prediction_std_normalized = float(np.std(neighbor_values))
            
            # Convert std to mmHg scale
            prediction_std_mmHg = prediction_std_normalized * self.target_std
            
            # Calculate 95% confidence interval (±1.96 * std)
            ci_lower = prediction_mmHg - 1.96 * prediction_std_mmHg
            ci_upper = prediction_mmHg + 1.96 * prediction_std_mmHg
            
        except Exception as e:
            prediction_std_normalized = 0.0
            prediction_std_mmHg = 0.0
            ci_lower = prediction_mmHg
            ci_upper = prediction_mmHg
        
        return {
            "predicted_normalized": float(prediction_normalized),
            "predicted_value_mmHg": float(prediction_mmHg),
            "confidence_interval_lower": float(ci_lower),
            "confidence_interval_upper": float(ci_upper),
            "prediction_std_normalized": prediction_std_normalized,
            "prediction_std_mmHg": prediction_std_mmHg
        }


class KNNDiastolicModel:
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None):
        """
        Initialize KNN Diastolic BP model
        
        Args:
            model_path: Path to the trained KNN model (.joblib)
            scaler_path: Path to the scaler (.pkl) for raw input normalization
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        
        # Denormalization constants
        self.target_mean = Config.DIASTOLIC_BP_MEAN
        self.target_std = Config.DIASTOLIC_BP_STD
        
        # Expected features (15 total, includes Systolic_BP)
        self.expected_features = [
            'Sex', 'Age', 'Height', 'Weight', 'Systolic_BP', 'Heart_Rate', 'BMI',
            'Diabetes_Diabetes', 'Diabetes_None', 'Diabetes_Type 2 Diabetes',
            'Cerebral_infarction_None', 'Cerebral_infarction_cerebral infarction',
            'Cerebrovascular_disease_None', 'Cerebrovascular_disease_cerebrovascular disease',
            'Cerebrovascular_disease_insufficiency of cerebral blood supply'
        ]
        
        # Numeric features that need scaling (6 features including Systolic_BP)
        self.numeric_features = ['Age', 'Height', 'Weight', 'Systolic_BP', 'Heart_Rate', 'BMI']
        
    def preprocess(self, input_data: Dict[str, Any], is_raw: bool = False) -> pd.DataFrame:
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Dictionary with feature values
            is_raw: If True, normalize numeric features using scaler
            
        Returns:
            DataFrame ready for model prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Ensure all expected features exist
        for feature in self.expected_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Reorder columns to match training data
        df = df[self.expected_features]
        
        # Normalize numeric features if raw input
        if is_raw and self.scaler:
            # Extract numeric features
            numeric_data = df[self.numeric_features].values
            
            # Scale numeric features
            scaled_numeric = self.scaler.transform(numeric_data)
            
            # Replace with scaled values
            df[self.numeric_features] = scaled_numeric
        
        return df
    
    def predict(self, input_data: Dict[str, Any], is_raw: bool = False) -> Dict[str, float]:
        """
        Predict Diastolic BP with denormalization
        
        Args:
            input_data: Dictionary with feature values
            is_raw: If True, input will be normalized before prediction
            
        Returns:
            Dictionary with:
                - predicted_normalized: Predicted value in normalized form
                - predicted_value_mmHg: Predicted value in mmHg (denormalized)
                - confidence_interval_lower: Lower bound of 95% confidence interval (mmHg)
                - confidence_interval_upper: Upper bound of 95% confidence interval (mmHg)
                - prediction_std_normalized: Std of prediction (normalized)
                - prediction_std_mmHg: Std of prediction (mmHg)
        """
        # Preprocess input
        processed_data = self.preprocess(input_data, is_raw)
        
        # Get prediction (normalized)
        prediction_normalized = self.model.predict(processed_data)[0]
        
        # Denormalize prediction to mmHg
        prediction_mmHg = prediction_normalized * self.target_std + self.target_mean
        
        # Calculate prediction uncertainty using k nearest neighbors
        try:
            # Get k nearest neighbors and their target values
            distances, indices = self.model.kneighbors(processed_data)
            
            # Get the target values of k nearest neighbors
            neighbor_values = self.model._y[indices[0]]
            
            # Calculate standard deviation of neighbor predictions (normalized)
            prediction_std_normalized = float(np.std(neighbor_values))
            
            # Convert std to mmHg scale
            prediction_std_mmHg = prediction_std_normalized * self.target_std
            
            # Calculate 95% confidence interval (±1.96 * std)
            ci_lower = prediction_mmHg - 1.96 * prediction_std_mmHg
            ci_upper = prediction_mmHg + 1.96 * prediction_std_mmHg
            
        except Exception as e:
            prediction_std_normalized = 0.0
            prediction_std_mmHg = 0.0
            ci_lower = prediction_mmHg
            ci_upper = prediction_mmHg
        
        return {
            "predicted_normalized": float(prediction_normalized),
            "predicted_value_mmHg": float(prediction_mmHg),
            "confidence_interval_lower": float(ci_lower),
            "confidence_interval_upper": float(ci_upper),
            "prediction_std_normalized": prediction_std_normalized,
            "prediction_std_mmHg": prediction_std_mmHg
        }
