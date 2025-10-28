from fastapi import APIRouter
from src.api.predict import predict_hypertension
from pydantic import BaseModel

router = APIRouter()

class PredictionRequest(BaseModel):
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

@router.post("/predict")
async def predict(request: PredictionRequest):
    return predict_hypertension(
        request.Sex,
        request.Age,
        request.Height,
        request.Weight,
        request.Systolic_BP,
        request.Diastolic_BP,
        request.Heart_Rate,
        request.BMI,
        request.Diabetes_Diabetes,
        request.Diabetes_None,
        request.Diabetes_Type2,
        request.Cerebral_infarction_None,
        request.Cerebral_infarction_infarction,
        request.Cerebrovascular_None,
        request.Cerebrovascular_disease,
        request.Cerebrovascular_insuff
    )