from fastapi import APIRouter, HTTPException, status
from utils.schemas import RandomForestPredictionRequest, PredictionResponse
from api.dependencies import get_random_forest_model
import pandas as pd


router = APIRouter(
    prefix="/random-forest",
    tags=["Random Forest"]
)


@router.post("/predict", response_model=PredictionResponse)
async def predict_hypertension(request: RandomForestPredictionRequest):
    """
    Predict hypertension using Random Forest model.
    
    Supports both RAW and NORMALIZED input:
    - RAW (is_raw=True): Age=45, Weight=70kg, Systolic_BP=130mmHg, ...
    - NORMALIZED (is_raw=False): Age=-0.768, Weight=0.237, Systolic_BP=1.626, ...
    """
    try:
        # Lấy model instance
        model = get_random_forest_model()
        
        # Chuyển đổi request thành DataFrame
        input_dict = request.dict(exclude={'is_raw'})  # Exclude is_raw từ data
        input_df = pd.DataFrame([input_dict])
        
        # Dự đoán với flag is_raw
        prediction, probability = model.predict(input_df, is_raw=request.is_raw)
        label = model.get_prediction_label(prediction)
        
        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability=probability,
            model_type="Random Forest"
        )
        
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

