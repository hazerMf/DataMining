from pydantic import BaseModel, Field
from typing import Optional


class RandomForestPredictionRequest(BaseModel):
    """
    Request schema cho Random Forest prediction
    
    Hỗ trợ CẢ 2 loại input:
    - RAW: Giá trị thực (Age=45, Weight=70kg, ...)
    - NORMALIZED: Giá trị đã chuẩn hóa (Age=-0.768, Weight=0.237, ...)
    """
    # Input type flag
    is_raw: bool = Field(
        default=False,
        description="True=RAW input (giá trị thực), False=NORMALIZED (z-score)"
    )
    
    # Numeric features
    Sex: int = Field(..., ge=0, le=1, description="Giới tính (0=Nữ, 1=Nam)")
    Age: float = Field(..., description="Tuổi (raw: 0-120, normalized: z-score)")
    Height: float = Field(..., description="Chiều cao (raw: cm, normalized: z-score)")
    Weight: float = Field(..., description="Cân nặng (raw: kg, normalized: z-score)")
    Systolic_BP: float = Field(..., description="Huyết áp tâm thu (raw: mmHg, normalized: z-score)")
    Diastolic_BP: float = Field(..., description="Huyết áp tâm trương (raw: mmHg, normalized: z-score)")
    Heart_Rate: float = Field(..., description="Nhịp tim (raw: bpm, normalized: z-score)")
    BMI: float = Field(..., description="Chỉ số BMI (raw: kg/m², normalized: z-score)")
    Diabetes_Diabetes: int = Field(..., description="Có tiểu đường (0/1)")
    Diabetes_None: int = Field(..., description="Không tiểu đường (0/1)")
    Diabetes_Type2: int = Field(..., description="Tiểu đường Type 2 (0/1)")
    Cerebral_infarction_None: int = Field(..., description="Không nhồi máu não (0/1)")
    Cerebral_infarction_infarction: int = Field(..., description="Có nhồi máu não (0/1)")
    Cerebrovascular_None: int = Field(..., description="Không bệnh mạch máu não (0/1)")
    Cerebrovascular_disease: int = Field(..., description="Có bệnh mạch máu não (0/1)")
    Cerebrovascular_insuff: int = Field(..., description="Thiếu máu não (0/1)")
    
    class Config:
        json_schema_extra = {
            "example_raw": {
                "is_raw": True,
                "Sex": 0,
                "Age": 45,
                "Height": 152,
                "Weight": 63,
                "Systolic_BP": 161,
                "Diastolic_BP": 89,
                "Heart_Rate": 97,
                "BMI": 27.27,
                "Diabetes_Diabetes": 0,
                "Diabetes_None": 1,
                "Diabetes_Type2": 0,
                "Cerebral_infarction_None": 1,
                "Cerebral_infarction_infarction": 0,
                "Cerebrovascular_None": 1,
                "Cerebrovascular_disease": 0,
                "Cerebrovascular_insuff": 0
            },
            "example_normalized": {
                "is_raw": False,
                "Sex": 0,
                "Age": -0.768337,
                "Height": -1.127587,
                "Weight": 0.236798,
                "Systolic_BP": 1.625816,
                "Diastolic_BP": 1.547085,
                "Heart_Rate": 2.180326,
                "BMI": 1.041960,
                "Diabetes_Diabetes": 0,
                "Diabetes_None": 1,
                "Diabetes_Type2": 0,
                "Cerebral_infarction_None": 1,
                "Cerebral_infarction_infarction": 0,
                "Cerebrovascular_None": 1,
                "Cerebrovascular_disease": 0,
                "Cerebrovascular_insuff": 0
            }
        }


class PredictionResponse(BaseModel):
    """Response schema cho predictions"""
    prediction: int = Field(..., description="Lớp dự đoán")
    label: str = Field(..., description="Nhãn có ý nghĩa")
    probability: float = Field(..., description="Xác suất dự đoán")
    model_type: str = Field(..., description="Loại model được sử dụng")


class KNNSystolicPredictionRequest(BaseModel):
    """
    Request schema for KNN Systolic BP prediction
    Requires Diastolic_BP as input feature
    """
    # Input type flag
    is_raw: bool = Field(
        default=False,
        description="True=RAW input (actual values), False=NORMALIZED (z-score)"
    )
    
    # Numeric features (6 need scaling, including Diastolic_BP)
    Sex: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    Age: float = Field(..., description="Age (raw: years, normalized: z-score)")
    Height: float = Field(..., description="Height (raw: cm, normalized: z-score)")
    Weight: float = Field(..., description="Weight (raw: kg, normalized: z-score)")
    Diastolic_BP: float = Field(..., description="Diastolic BP (raw: mmHg, normalized: z-score) - REQUIRED for SBP prediction")
    Heart_Rate: float = Field(..., description="Heart rate (raw: bpm, normalized: z-score)")
    BMI: float = Field(..., description="BMI (raw: kg/m², normalized: z-score)")
    
    # Categorical features (one-hot encoded, already 0/1)
    Diabetes_Diabetes: int = Field(default=0, ge=0, le=1, description="Has diabetes (0/1)")
    Diabetes_None: int = Field(default=1, ge=0, le=1, description="No diabetes (0/1)")
    Diabetes_Type2: int = Field(default=0, ge=0, le=1, description="Type 2 diabetes (0/1)")
    Cerebral_infarction_None: int = Field(default=1, ge=0, le=1, description="No cerebral infarction (0/1)")
    Cerebral_infarction_infarction: int = Field(default=0, ge=0, le=1, description="Has cerebral infarction (0/1)")
    Cerebrovascular_None: int = Field(default=1, ge=0, le=1, description="No cerebrovascular disease (0/1)")
    Cerebrovascular_disease: int = Field(default=0, ge=0, le=1, description="Has cerebrovascular disease (0/1)")
    Cerebrovascular_insuff: int = Field(default=0, ge=0, le=1, description="Cerebral blood insufficiency (0/1)")
    
    class Config:
        json_schema_extra = {
            "example_raw": {
                "is_raw": True,
                "Sex": 0,
                "Age": 45,
                "Height": 152,
                "Weight": 63,
                "Diastolic_BP": 89,
                "Heart_Rate": 97,
                "BMI": 27.27,
                "Diabetes_Diabetes": 0,
                "Diabetes_None": 1,
                "Diabetes_Type2": 0,
                "Cerebral_infarction_None": 1,
                "Cerebral_infarction_infarction": 0,
                "Cerebrovascular_None": 1,
                "Cerebrovascular_disease": 0,
                "Cerebrovascular_insuff": 0
            },
            "example_normalized": {
                "is_raw": False,
                "Sex": 0,
                "Age": -0.768337,
                "Height": -1.127587,
                "Weight": 0.236798,
                "Diastolic_BP": 1.547085,
                "Heart_Rate": 2.180326,
                "BMI": 1.041960,
                "Diabetes_Diabetes": 0,
                "Diabetes_None": 1,
                "Diabetes_Type2": 0,
                "Cerebral_infarction_None": 1,
                "Cerebral_infarction_infarction": 0,
                "Cerebrovascular_None": 1,
                "Cerebrovascular_disease": 0,
                "Cerebrovascular_insuff": 0
            }
        }


class KNNDiastolicPredictionRequest(BaseModel):
    """
    Request schema for KNN Diastolic BP prediction
    Requires Systolic_BP as input feature
    """
    # Input type flag
    is_raw: bool = Field(
        default=False,
        description="True=RAW input (actual values), False=NORMALIZED (z-score)"
    )
    
    # Numeric features (6 need scaling, including Systolic_BP)
    Sex: int = Field(..., ge=0, le=1, description="Gender (0=Female, 1=Male)")
    Age: float = Field(..., description="Age (raw: years, normalized: z-score)")
    Height: float = Field(..., description="Height (raw: cm, normalized: z-score)")
    Weight: float = Field(..., description="Weight (raw: kg, normalized: z-score)")
    Systolic_BP: float = Field(..., description="Systolic BP (raw: mmHg, normalized: z-score) - REQUIRED for DBP prediction")
    Heart_Rate: float = Field(..., description="Heart rate (raw: bpm, normalized: z-score)")
    BMI: float = Field(..., description="BMI (raw: kg/m², normalized: z-score)")
    
    # Categorical features (one-hot encoded, already 0/1)
    Diabetes_Diabetes: int = Field(default=0, ge=0, le=1, description="Has diabetes (0/1)")
    Diabetes_None: int = Field(default=1, ge=0, le=1, description="No diabetes (0/1)")
    Diabetes_Type2: int = Field(default=0, ge=0, le=1, description="Type 2 diabetes (0/1)")
    Cerebral_infarction_None: int = Field(default=1, ge=0, le=1, description="No cerebral infarction (0/1)")
    Cerebral_infarction_infarction: int = Field(default=0, ge=0, le=1, description="Has cerebral infarction (0/1)")
    Cerebrovascular_None: int = Field(default=1, ge=0, le=1, description="No cerebrovascular disease (0/1)")
    Cerebrovascular_disease: int = Field(default=0, ge=0, le=1, description="Has cerebrovascular disease (0/1)")
    Cerebrovascular_insuff: int = Field(default=0, ge=0, le=1, description="Cerebral blood insufficiency (0/1)")
    
    class Config:
        json_schema_extra = {
            "example_raw": {
                "is_raw": True,
                "Sex": 0,
                "Age": 45,
                "Height": 152,
                "Weight": 63,
                "Systolic_BP": 161,
                "Heart_Rate": 97,
                "BMI": 27.27,
                "Diabetes_Diabetes": 0,
                "Diabetes_None": 1,
                "Diabetes_Type2": 0,
                "Cerebral_infarction_None": 1,
                "Cerebral_infarction_infarction": 0,
                "Cerebrovascular_None": 1,
                "Cerebrovascular_disease": 0,
                "Cerebrovascular_insuff": 0
            },
            "example_normalized": {
                "is_raw": False,
                "Sex": 0,
                "Age": -0.768337,
                "Height": -1.127587,
                "Weight": 0.236798,
                "Systolic_BP": 1.625816,
                "Heart_Rate": 2.180326,
                "BMI": 1.041960,
                "Diabetes_Diabetes": 0,
                "Diabetes_None": 1,
                "Diabetes_Type2": 0,
                "Cerebral_infarction_None": 1,
                "Cerebral_infarction_infarction": 0,
                "Cerebrovascular_None": 1,
                "Cerebrovascular_disease": 0,
                "Cerebrovascular_insuff": 0
            }
        }


class BPPredictionResponse(BaseModel):
    """
    Enhanced Response schema for Blood Pressure predictions with denormalization
    """
    # Normalized values (from model output)
    predicted_normalized: float = Field(..., description="Predicted BP in normalized form (z-score)")
    prediction_std_normalized: float = Field(..., description="Std deviation of k-neighbors in normalized form")
    
    # Real values (denormalized to mmHg)
    predicted_value_mmHg: float = Field(..., description="Predicted BP in mmHg (denormalized)")
    prediction_std_mmHg: float = Field(..., description="Std deviation in mmHg")
    
    # Confidence interval (95% CI)
    confidence_interval_lower: float = Field(..., description="Lower bound of 95% confidence interval (mmHg)")
    confidence_interval_upper: float = Field(..., description="Upper bound of 95% confidence interval (mmHg)")
    
    # Metadata
    input_type: str = Field(..., description="Type of input used (raw/normalized)")
    model_type: str = Field(..., description="Model used (knn_systolic/knn_diastolic)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predicted_normalized": 0.3738,
                "prediction_std_normalized": 0.7373,
                "predicted_value_mmHg": 135.6,
                "prediction_std_mmHg": 15.02,
                "confidence_interval_lower": 106.2,
                "confidence_interval_upper": 165.0,
                "input_type": "raw",
                "model_type": "knn_systolic"
            }
        }


class HealthCheckResponse(BaseModel):
    """Response cho health check endpoint"""
    status: str
    message: str
    models: dict


class ErrorResponse(BaseModel):
    """Response cho errors"""
    detail: str
    error_type: Optional[str] = None


# Legacy schemas 
class HypertensionInput(BaseModel):
    Sex: int
    Age: float
    Height: float
    Weight: float
    Systolic_BP: float
    Diastolic_BP: float
    Heart_Rate: float
    BMI: float
    Diabetes_Diabetes: bool
    Diabetes_None: bool
    Diabetes_Type2: bool
    Cerebral_infarction_None: bool
    Cerebral_infarction_infarction: bool
    Cerebrovascular_None: bool
    Cerebrovascular_disease: bool
    Cerebrovascular_insuff: bool

class HypertensionPrediction(BaseModel):
    label: str
    probability: float