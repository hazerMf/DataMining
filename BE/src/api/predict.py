from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.models.model import load_model, make_prediction

router = APIRouter()

class InputData(BaseModel):
    Sex: int
    Age: float
    Height: float
    Weight: float
    Systolic_BP: float
    Diastolic_BP: float
    Heart_Rate: float
    BMI: float
    Diabetes_Diabetes: int
    Diabetes_None: int
    Diabetes_Type2: int
    Cerebral_infarction_None: int
    Cerebral_infarction_infarction: int
    Cerebrovascular_None: int
    Cerebrovascular_disease: int
    Cerebrovascular_insuff: int

model = load_model()

@router.post("/predict")
def predict(input_data: InputData):
    input_df = pd.DataFrame([input_data.dict()])
    prediction, probability = make_prediction(model, input_df)
    
    label_map = {
        0: "Bình thường (Normal)",
        1: "Tiền tăng huyết áp (Prehypertension)",
        2: "Tăng huyết áp giai đoạn 1 (Stage 1 Hypertension)",
        3: "Tăng huyết áp giai đoạn 2 (Stage 2 Hypertension)"
    }
    
    label = label_map.get(prediction, "Không xác định")
    
    return {
        "result": label,
        "probability": probability
    }