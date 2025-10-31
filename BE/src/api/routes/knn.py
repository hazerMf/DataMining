from fastapi import APIRouter, HTTPException, Depends
from src.utils.schemas import (
    KNNSystolicPredictionRequest,
    KNNDiastolicPredictionRequest,
    BPPredictionResponse,
    ErrorResponse
)
from src.api.dependencies import get_knn_systolic_model, get_knn_diastolic_model

router = APIRouter(prefix="/knn", tags=["KNN Blood Pressure Prediction"])


@router.post(
    "/predict/systolic",
    response_model=BPPredictionResponse,
    summary="Predict Systolic Blood Pressure",
    description="""
    Dự đoán Systolic BP.
    """
)
async def predict_systolic_bp(
    request: KNNSystolicPredictionRequest,
    model = Depends(get_knn_systolic_model)
):
    """
    Predict Systolic Blood Pressure using KNN regression
    
    Args:
        request: Input features including Diastolic_BP
        model: KNN Systolic model instance
        
    Returns:
        BPPredictionResponse with predicted Systolic BP value
    """
    try:
        # Convert request to dict
        input_data = request.model_dump()
        is_raw = input_data.pop('is_raw', False)
        
        # Map field names to match model expectations
        if 'Diabetes_Type2' in input_data:
            input_data['Diabetes_Type 2 Diabetes'] = input_data.pop('Diabetes_Type2')
        if 'Cerebral_infarction_infarction' in input_data:
            input_data['Cerebral_infarction_cerebral infarction'] = input_data.pop('Cerebral_infarction_infarction')
        if 'Cerebrovascular_disease' in input_data:
            input_data['Cerebrovascular_disease_cerebrovascular disease'] = input_data.pop('Cerebrovascular_disease')
        if 'Cerebrovascular_insuff' in input_data:
            input_data['Cerebrovascular_disease_insufficiency of cerebral blood supply'] = input_data.pop('Cerebrovascular_insuff')
        if 'Cerebrovascular_None' in input_data:
            input_data['Cerebrovascular_disease_None'] = input_data.pop('Cerebrovascular_None')
        
        # Make prediction
        result = model.predict(input_data, is_raw=is_raw)
        
        return BPPredictionResponse(
            predicted_normalized=result["predicted_normalized"],
            prediction_std_normalized=result["prediction_std_normalized"],
            predicted_value_mmHg=result["predicted_value_mmHg"],
            prediction_std_mmHg=result["prediction_std_mmHg"],
            confidence_interval_lower=result["confidence_interval_lower"],
            confidence_interval_upper=result["confidence_interval_upper"],
            input_type="raw" if is_raw else "normalized",
            model_type="knn_systolic"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@router.post(
    "/predict/diastolic",
    response_model=BPPredictionResponse,
    summary="Predict Diastolic Blood Pressure",
    description="""
    Dự đoán Diastolic BP s
    """
)
async def predict_diastolic_bp(
    request: KNNDiastolicPredictionRequest,
    model = Depends(get_knn_diastolic_model)
):
    """
    Predict Diastolic Blood Pressure using KNN regression
    
    Args:
        request: Input features including Systolic_BP
        model: KNN Diastolic model instance
        
    Returns:
        BPPredictionResponse with predicted Diastolic BP value
    """
    try:
        # Convert request to dict
        input_data = request.model_dump()
        is_raw = input_data.pop('is_raw', False)
        
        # Map field names to match model expectations
        if 'Diabetes_Type2' in input_data:
            input_data['Diabetes_Type 2 Diabetes'] = input_data.pop('Diabetes_Type2')
        if 'Cerebral_infarction_infarction' in input_data:
            input_data['Cerebral_infarction_cerebral infarction'] = input_data.pop('Cerebral_infarction_infarction')
        if 'Cerebrovascular_disease' in input_data:
            input_data['Cerebrovascular_disease_cerebrovascular disease'] = input_data.pop('Cerebrovascular_disease')
        if 'Cerebrovascular_insuff' in input_data:
            input_data['Cerebrovascular_disease_insufficiency of cerebral blood supply'] = input_data.pop('Cerebrovascular_insuff')
        if 'Cerebrovascular_None' in input_data:
            input_data['Cerebrovascular_disease_None'] = input_data.pop('Cerebrovascular_None')
        
        # Make prediction
        result = model.predict(input_data, is_raw=is_raw)
        
        return BPPredictionResponse(
            predicted_normalized=result["predicted_normalized"],
            prediction_std_normalized=result["prediction_std_normalized"],
            predicted_value_mmHg=result["predicted_value_mmHg"],
            prediction_std_mmHg=result["prediction_std_mmHg"],
            confidence_interval_lower=result["confidence_interval_lower"],
            confidence_interval_upper=result["confidence_interval_upper"],
            input_type="raw" if is_raw else "normalized",
            model_type="knn_diastolic"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
