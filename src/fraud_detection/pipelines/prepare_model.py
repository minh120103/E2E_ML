from src.fraud_detection.config.configuration import ConfigurationManager
from src.fraud_detection.components.base_model import PrepareBaseModel
from src.fraud_detection import logger
# import mlflow
# import dagshub
STAGE_NAME = "Prepare base model"


class ModelPreparationPipeline:
    def __init__(self):
        # self.mlflow_config = mlflow_config
        pass
    def main(self, X_train, X_test):
        logger.info(f">>> Stage {STAGE_NAME} started <<<")
        prepare_base_model_config = ConfigurationManager().get_prepare_base_model_config()
        
        # mlflow.log_params({
        #     "n_estimators": prepare_base_model_config.n_estimators,
        #     "random_state": prepare_base_model_config.random_state,
        #     "criterion": prepare_base_model_config.criterion,
        #     "max_depth": prepare_base_model_config.max_depth,
        #     "max_features": prepare_base_model_config.max_features,
        #     "min_samples_leaf": prepare_base_model_config.min_samples_leaf
        # })
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
        obj = ModelPreparationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e