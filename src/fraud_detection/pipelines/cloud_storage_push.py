from src.fraud_detection.config.configuration import ConfigurationManager
from src.fraud_detection.components.cloud_storage_push import CloudStoragePush
from src.fraud_detection import logger

STAGE_NAME = "Cloud Storage Push"



class CloudStoragePushPipeline:
    def main(self):
        logger.info(f">>> Stage {STAGE_NAME} started <<<")
        
        config = ConfigurationManager()
        cloud_storage_push_config = config.get_cloud_storage_push_config()
        cloud_storage_push = CloudStoragePush(config=cloud_storage_push_config)
        cloud_storage_push.push_to_cloud_storage()
        
        logger.info(f">>> Stage {STAGE_NAME} completed <<<")