
from fastapi import APIRouter, HTTPException, status
from utils.schemas import Model2PredictionRequest, PredictionResponse
from api.dependencies import get_model_2
import pandas as pd


router = APIRouter(
    prefix="/model-2",
    tags=["Model 2 - Prediction"]
)


@router.post("/predict", response_model=PredictionResponse)
async def predict_model_2(request: Model2PredictionRequest):
    """
    Dự đoán sử dụng Model 2
    
    TODO: Cập nhật description và logic
    
    - **Returns**: Kết quả dự đoán
    """
    try:
        # Lấy model instance
        model = get_model_2()
        
        # Chuyển đổi request thành DataFrame
        input_dict = request.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Dự đoán
        prediction, probability = model.predict(input_df)
        
        # TODO: Thay thế label logic
        label = f"Prediction class {prediction}"
        
        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability=probability,
            model_type="Model 2"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


