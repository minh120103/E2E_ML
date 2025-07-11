from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.fraud_detection.pipelines.predict_pipeline import ChurnController

router = APIRouter(
    prefix="/fraud",
    tags=["Fraud Prediction"],
    responses={404: {"description": "Not found"}},
)

class FraudResponse(BaseModel):
    message: str
    s3_url: Optional[str] = None

@router.post("/", response_model=FraudResponse)
async def predict_fraud(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_version: str = Form(default="1"),
    scaler_version: str = Form(default="scaler/scaler_churn_version_20250708T142954.pkl"),
    run_id: str = Form(default="621b0660b6674e21b5f7090fbc211a6f"),
    model_name: str = Form(default="XGBoost"),
):
    return await ChurnController.predict_fraud(background_tasks=background_tasks, file=file, model_version=model_version, scaler_version=scaler_version, run_id=run_id, model_name=model_name)