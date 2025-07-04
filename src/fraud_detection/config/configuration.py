from src.fraud_detection.constants import *
from src.fraud_detection.utils.common import read_yaml, create_directories
from src.fraud_detection.entity.config_entity import DataIngestionConfig, PrepareBaseModelConfig, TrainingConfig
from src.fraud_detection import logger

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir, config.data_version_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            test_size=config.test_size,
            random_state=config.random_state,
            data_version_dir=Path(config.data_version_dir),
            bucket_name=config.bucket_name
            )

        # logger.info(f"Data Ingestion config: {config}")
        return data_ingestion_config
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.model_version_dir, config.data_version_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
            n_estimators=config.n_estimators,
            random_state=config.random_state,
            learning_rate=config.learning_rate,
            max_depth=config.max_depth,
            scale_pos_weight=config.scale_pos_weight,
            n_jobs=config.n_jobs,
            objective=config.objective
        )

        logger.info(f"Prepare base model config: {config}")
        return prepare_base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        config = self.config.training
        
        create_directories([config.model_version_dir, config.data_version_dir])

        training_config = TrainingConfig(
            model_version_dir=Path(config.model_version_dir),
            data_version_dir=Path(config.data_version_dir),
        )

        logger.info(f"Training config: {config}")
        return training_config