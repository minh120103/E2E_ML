import os
from src.fraud_detection.config.configuration import ConfigurationManager
from src.fraud_detection.pipelines.preprocess_data import DataPreparationPipeline
from .prepare_model import ModelPreparationPipeline
from .train_evaluation import TrainEvaluationPipeline
from .cloud_storage_push import CloudStoragePushPipeline
from src.fraud_detection import logger
from src.fraud_detection.pipelines.clean_up import cleanup_temp_files
from src.fraud_detection.components.support import import_data
from pathlib import Path
from fastapi import UploadFile
import mlflow
import dagshub
from datetime import datetime

class WorkflowRunner:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.uploaded_file = None

    def check_data_file_exists(self):
        """Check if the local data file exists and has content"""
        try:
            config = self.config_manager.get_data_ingestion_config()
            data_file_path = Path(config.local_data_file)
            
            if data_file_path.exists() and data_file_path.stat().st_size > 0:
                logger.info(f"Data file found: {data_file_path}")
                return True
            else:
                logger.info(f"Data file not found or empty: {data_file_path}")
                return False
        except Exception as e:
            logger.error(f"Error checking data file: {e}")
            return False
            
    async def run(self, uploaded_file: UploadFile = None):
        """Run the complete workflow with proper path passing between stages."""
        self.uploaded_file = uploaded_file
        
        try:
            mlflow_config = self.config_manager.get_mlflow_config()
            logger.info(f"MLflow configured with experiment: {mlflow_config.experiment_name}")
            dagshub.init(
                repo_owner=mlflow_config.dagshub_username,
                repo_name=mlflow_config.dagshub_repo_name,
                mlflow=True
            )
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            mlflow.set_experiment(mlflow_config.experiment_name)
        except Exception as e:
            logger.warning(f"MLflow configuration failed: {e}. Continuing without MLflow tracking.")
            mlflow_config = None

        if not self.check_data_file_exists():
            if self.uploaded_file is not None:
                logger.info("=" * 50)
                logger.info("STAGE 0: Data Import")
                logger.info("=" * 50)
                
                try:
                    await import_data(self.uploaded_file)
                    logger.info("Data import completed successfully")
                except Exception as e:
                    logger.error(f"Data import failed: {e}")
                    raise
            else:
                raise ValueError("No data file found and no uploaded file provided. Please provide a data file.")

        # Stage 1: Data Preparation
        try:
            logger.info(f">>>>>> stage Data Ingestion started <<<<<<") 
            data_ingestion_pipeline = DataPreparationPipeline()
            X_train, X_test, y_train, y_test = data_ingestion_pipeline.main()
            logger.info(f">>>>>> stage Data Ingestion completed <<<<<<\n\nx====================x")
        except Exception as e:
                logger.exception(e)
                raise e
        # Stage 2: Model Preparation
        try:
            logger.info(f">>>>>> stage Model Training Pipeline started <<<<<<")

            # Initialize Configuration Manager to get MLflow config
            config_manager = ConfigurationManager()
            mlflow_config = config_manager.get_mlflow_config()

            # Initialize DagsHub for MLflow tracking
            os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
            dagshub.init(repo_owner=mlflow_config.dagshub_username, repo_name=mlflow_config.dagshub_repo_name, mlflow=True)
            mlflow.set_tracking_uri(mlflow_config.tracking_uri)
            mlflow.set_experiment(mlflow_config.experiment_name)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"fraud_model_training_cycle_{timestamp}"

            with mlflow.start_run(run_name=run_name):
                logger.info("=" * 50)
                logger.info("Running Model Preparation")
                model_prep_pipeline = ModelPreparationPipeline(mlflow_config=mlflow_config)
                model, base_model_path, scaler_path, X_train_scaled, X_test_scaled = model_prep_pipeline.main(
                    X_train=X_train,
                    X_test=X_test
                )
                logger.info(f"Base model created at: {base_model_path}")
                logger.info(f"Scaler created at: {scaler_path}")
                logger.info("=" * 50)

                logger.info("=" * 50)
                logger.info("Running Model Training and Evaluation")
                train_eval_pipeline = TrainEvaluationPipeline(mlflow_config=mlflow_config)
                final_model, metrics, final_model_path = train_eval_pipeline.main(
                    base_model=model,
                    X_train_scaled=X_train_scaled,
                    X_test_scaled=X_test_scaled,
                    y_train=y_train,
                    y_test=y_test
                )
                logger.info(f"Training and evaluation completed. Final model at: {final_model_path}")
                logger.info(f"Logged Metrics: {metrics}")
                logger.info("=" * 50)

            logger.info(f">>>>>> stage Model Training Pipeline completed <<<<<<\n\nx====================x")

            logger.info(f">>>>>> stage STAGE 4: Cloud Storage Push started <<<<<<") 
            logger.info("=" * 50)
            logger.info("STAGE 4: Cloud Storage Push")
            logger.info("=" * 50)
            
            cloud_push = CloudStoragePushPipeline()
            cloud_push.main()
            logger.info("Cloud storage push completed successfully")
            logger.info(f">>>>>> stage STAGE 4: Cloud Storage Push completed <<<<<<\n\nx====================x")
            cleanup_temp_files()
            # logger.info("Temporary files cleaned up successfully")
        except Exception as e:
            logger.exception(e)
            raise e
        return final_model_path