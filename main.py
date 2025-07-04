from src.fraud_detection import logger
from src.fraud_detection.pipelines.preprocess_data import DataPreparationPipeline
from src.fraud_detection.pipelines.prepare_model import ModelPreparationPipeline

logger.info("Starting the fraud detection application...")

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataPreparationPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


logger.info("Starting the fraud detection application...")

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   model_prepare = ModelPreparationPipeline()
   model_prepare.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e