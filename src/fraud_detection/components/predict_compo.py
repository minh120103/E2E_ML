import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from src.fraud_detection.components.data_ingestion import DataIngestion
from src.fraud_detection.config.configuration import ConfigurationManager, WebhookConfig
import joblib 
import mlflow
from src.fraud_detection import logger
# from src.fraud_detection.utils.notify_webhook import post_to_webhook
# from src.fraud_detection.utils.visualize_ouput import visualize_customer_churn
from datetime import datetime
import time
import os
import dagshub
import tempfile
import os
from src.fraud_detection.components.cloud_storage_push import CloudStoragePush # Import CloudStoragePush
# web_hook_url = WebhookConfig().url

# async def send_webhook_payload(
#     message: str,
#     avg_confidence: Optional[float] = None,
# ):
#     try:
#         logger.info("Preparing webhook notification")

#         payload = {
#             "message": message,
#             "avg_confidence": avg_confidence,
#         }

#         await post_to_webhook(web_hook_url, payload)

#     except Exception as e:
#         logger.error(f"Webhook notification failed with unexpected error: {str(e)}")
class PredictionPipeline:
    def __init__(self, model_uri: str, scaler_uri: str):
        try:
            self.model = mlflow.pyfunc.load_model(model_uri)
            scaler_path = mlflow.artifacts.download_artifacts(artifact_uri=scaler_uri)
            self.scaler = joblib.load(scaler_path)
        except Exception as e:
                raise RuntimeError(f"Failed to load model or scaler: {e}")
    def process_data_for_churn(self,df_input):
        df_input['MoneySpentSender'] = df_input['oldbalanceOrg'] - df_input['newbalanceOrig']
        df_input['MoneyReceiver'] = df_input['oldbalanceDest'] - df_input['newbalanceDest']
        df_input = df_input.drop(columns=['newbalanceOrig'])
        df_input = df_input.drop(columns=['newbalanceDest'])
        df_input = df_input.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
        df_input = pd.get_dummies(df_input, columns=['type'], dtype=int)

        return df_input
    async def predict(self):
        try:
            start_time = time.time()
            start_datetime = datetime.now()
            time_str = start_datetime.strftime('%Y%m%dT%H%M%S')
            
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()

            with mlflow.start_run(run_name=f"prediction_run_{time_str}"):
                data_ingestion = DataIngestion(config=data_ingestion_config)
                
                df = data_ingestion.load_data()
                df_features = self.process_data_for_churn(df)
                X = self.scaler.transform(df_features)

                y_pred = self.model.predict(X)
                df_features['isFraud'] = y_pred
                counts = df_features['isFraud'].value_counts()
                count_fraud = counts.get(1, 0)
                count_not_fraud = counts.get(0, 0)               
                try:
                    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                        prediction_csv_path = temp_file.name
                        df_features.to_csv(prediction_csv_path, index=False)
                        mlflow.log_artifact(prediction_csv_path, "predictions")
                        logger.info(f"Successfully saved prediction results to {prediction_csv_path} and logged as MLflow artifact")                
                    config = ConfigurationManager()
                    cloud_storage_config = config.get_cloud_storage_push_config()
                    cloud_storage_push = CloudStoragePush(config=cloud_storage_config)
                    s3_url = cloud_storage_push.upload_prediction_to_s3(prediction_csv_path)
                    os.remove(prediction_csv_path)
                    logger.info(f"Deleted temporary prediction file: {prediction_csv_path}")
                except Exception as e:
                    logger.error(f"An error occurred during prediction saving, upload, or cleanup: {e}")
                
                try:    
                    sklearn_model = self.model._model_impl  
                    y_proba = sklearn_model.predict_proba(X)
                    max_confidence = y_proba.max(axis=1)
                    average_confidence = max_confidence.mean()
                except AttributeError:
                    average_confidence = None 
                end_time = time.time()
                end_datetime = datetime.now()
                processing_time = end_time - start_time

                logger.info(f"Prediction processing time: {processing_time:.2f} seconds")
                logger.info(f"Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Completed at: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                # plot_path = visualize_customer_churn(df_features)
                # mlflow.log_artifact(plot_path, "visualization")
                # os.remove(plot_path)
                mlflow.log_metric("processing_time_seconds", processing_time)
                mlflow.log_metric("count_fraud", count_fraud)
                mlflow.log_metric("count_not_fraud", count_not_fraud)
                mlflow.log_param("start_time", start_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("end_time", end_datetime.strftime('%Y-%m-%d %H:%M:%S'))
                mlflow.log_param("rawdata_records", len(df))
                mlflow.log_metric("records_processed", len(df_features))
                
                message = "Prediction task started in background. Results will be saved to experiment in MLflow."
                
                CONFIDENCE_THRESHOLD = 0.7
                if average_confidence is not None:
                    mlflow.log_metric("average_prediction_confidence", average_confidence)
                    
                    if average_confidence < CONFIDENCE_THRESHOLD:
                        message = (
                            f"⚠️ Average prediction confidence ({average_confidence:.2%}) is below the threshold "
                            f"of {CONFIDENCE_THRESHOLD:.2%}. Consider retraining the model."
                        )
                    else:
                        message = (
                            f"Average prediction confidence ({average_confidence:.2%}) is above the threshold "
                            f"of {CONFIDENCE_THRESHOLD:.2%}. No further action required."
                        )
                    
                    # await send_webhook_payload(message=message, avg_confidence=average_confidence)
                
                mlflow.log_text(message, "prediction_summary.txt")
            return message, s3_url

        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")
        
async def run_prediction_task(
    file_path: str,
    model_version: str,
    scaler_version: str,
    run_id: str,
    model_name: str = "XGBoost",
):
    """
    Background task to run prediction pipeline
    """
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        scaler_uri = f"runs:/{run_id}/{scaler_version}"
        pipeline = PredictionPipeline(model_uri, scaler_uri)
        message, s3_url = await pipeline.predict()

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleanup: Deleted input file {file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete input file during cleanup: {e}")

        return message, s3_url

    except Exception as e:
        logger.error(f"Background prediction task error: {e}")
        return f"Prediction error: {e}"