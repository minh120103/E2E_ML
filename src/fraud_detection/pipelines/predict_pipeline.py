from fastapi import UploadFile, HTTPException,Form,BackgroundTasks
from src.fraud_detection.components.support import import_data
from dotenv import load_dotenv
load_dotenv()
from src.fraud_detection.components.data_ingestion import DataIngestion
from src.fraud_detection.config.configuration import ConfigurationManager, WebhookConfig
from src.fraud_detection.components.predict_compo import run_prediction_task
from src.fraud_detection import logger
# from src.fraud_detection.utils.notify_webhook import post_to_webhook
# from src.fraud_detection.utils.visualize_ouput import visualize_customer_churn


class ChurnController:
    @staticmethod
    async def predict_fraud(
        background_tasks: BackgroundTasks,
        file: UploadFile,
        model_version: str = Form(default="1"),
        scaler_version: str = Form(default="scaler_churn_version_20250708T142954.pkl"),
        run_id: str = Form(default="621b0660b6674e21b5f7090fbc211a6f"),
        model_name: str = Form(default="XGBoost"),
    ):
        """
        Predict churn using uploaded file and dynamic model/scaler versions.
        """
        if not file:
            raise HTTPException(status_code=400, detail="No file uploaded.")

        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        input_file_path = data_ingestion_config.local_data_file

        try:
            await import_data(file)


            message, s3_url = await run_prediction_task(
                file_path=input_file_path,
                model_version=model_version,
                scaler_version=scaler_version,
                run_id=run_id,
                model_name=model_name
            )


            return {
                "message": message,
                "s3_url": s3_url,
            }

        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")