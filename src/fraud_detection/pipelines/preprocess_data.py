from src.fraud_detection.config.configuration import ConfigurationManager
from src.fraud_detection.components.data_ingestion import DataIngestion
from src.fraud_detection import logger


STAGE_NAME = "Data Ingestion stage"
class DataPreparationPipeline:
    def main(self):
        logger.info(f">>> Stage {STAGE_NAME} started <<<")
        
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        X_train, X_test, y_train, y_test,_,_ = data_ingestion.data_ingestion_pipeline()
        
        logger.info(f">>> Stage {STAGE_NAME} completed <<<")
        return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e