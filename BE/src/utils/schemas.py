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


class Model2PredictionRequest(BaseModel):
    """
    Request schema cho Model 2 prediction
    """
    feature_1: float = Field(..., description="Feature 1")
    feature_2: float = Field(..., description="Feature 2")
    
    class Config:
        schema_extra = {
            "example": {
                "feature_1": 0.5,
                "feature_2": 0.6,
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