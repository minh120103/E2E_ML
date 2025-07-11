from src.fraud_detection.config.configuration import ConfigurationManager
from src.fraud_detection.components.base_model import PrepareBaseModel
from src.fraud_detection.pipelines.preprocess_data import DataPreparationPipeline

from src.fraud_detection import logger
import mlflow
import dagshub
STAGE_NAME = "Prepare base model"


class ModelPreparationPipeline:
    def __init__(self, mlflow_config):
        self.mlflow_config = mlflow_config

    def main(self, X_train, X_test):
        logger.info(f">>> Stage {STAGE_NAME} started <<<")
        prepare_base_model_config = ConfigurationManager().get_prepare_base_model_config()
        
        mlflow.log_params({
            "n_estimators": prepare_base_model_config.n_estimators,
            "learning_rate": prepare_base_model_config.learning_rate,
            "random_state": prepare_base_model_config.random_state,
            "max_depth": prepare_base_model_config.max_depth,
            "scale_pos_weight": prepare_base_model_config.scale_pos_weight,
            "objective": prepare_base_model_config.objective
        })
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        model, base_model_path, scaler_path, X_train_scaled, X_test_scaled = prepare_base_model.full_model(
            X_train=X_train,
            X_test=X_test,
        )
        
        logger.info(f">>> Stage {STAGE_NAME} completed <<<")
        return model, base_model_path, scaler_path, X_train_scaled, X_test_scaled
    
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        data_pipeline = DataPreparationPipeline()
        X_train, X_test, _, _ = data_pipeline.main()
        
        # Now we can run the model preparation pipeline
        config_manager = ConfigurationManager()
        mlflow_config = config_manager.get_mlflow_config()
        model_prep_pipeline = ModelPreparationPipeline(mlflow_config=mlflow_config)
        model_prep_pipeline.main(X_train, X_test)
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e