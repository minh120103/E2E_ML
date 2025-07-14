from src.fraud_detection import logger
from src.fraud_detection.config.configuration import ConfigurationManager
from src.fraud_detection.pipelines.preprocess_data import DataPreparationPipeline
from src.fraud_detection.pipelines.prepare_model import ModelPreparationPipeline
from src.fraud_detection.pipelines.train_evaluation import TrainEvaluationPipeline
from src.fraud_detection.pipelines.cloud_storage_push import CloudStoragePushPipeline
from src.fraud_detection.pipelines.clean_up import cleanup_temp_files
import mlflow
import dagshub
from datetime import datetime

# STAGE 1: Data Ingestion & Preparation
STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion_pipeline = DataPreparationPipeline()
   X_train, X_test, y_train, y_test = data_ingestion_pipeline.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx====================x")
except Exception as e:
        logger.exception(e)
        raise e

# STAGE 2 & 3: Model Preparation, Training, and Evaluation with MLflow
STAGE_NAME = "Model Training Pipeline"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

    # Initialize Configuration Manager to get MLflow config
    config_manager = ConfigurationManager()
    mlflow_config = config_manager.get_mlflow_config()

    # Initialize DagsHub for MLflow tracking
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

    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx====================x")

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
