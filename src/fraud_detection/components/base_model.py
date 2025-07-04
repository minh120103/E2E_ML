import os
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
import joblib as jb
import pandas as pd
from datetime import datetime
from src.fraud_detection import logger
from src.fraud_detection.entity.config_entity import PrepareBaseModelConfig
# import mlflow

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')
        
    def get_base_model(self):
        """Create and return a base XGBoost model for churn prediction."""
        logger.info("Creating base XGBoost model")

        model = XGBClassifier(
            n_estimators=self.config.n_estimators, 
            random_state=self.config.random_state,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            scale_pos_weight=self.config.scale_pos_weight,
            objective=self.config.criterion,
            n_jobs=self.config.n_jobs,
        )
        logger.info(f"Model params:{self.config.n_estimators}, {self.config.random_state}, {self.config.learning_rate}, {self.config.max_depth}, {self.config.scale_pos_weight}, {self.config.n_jobs}")
        return model
    
    def scaler(self, X_train, X_test):
        """Prepare and save scaler based on training data."""
        logger.info("Creating scaler from training data")
        
        os.makedirs(self.config.data_version_dir, exist_ok=True)
        os.makedirs(self.config.model_version_dir, exist_ok=True)

        scaler_path = os.path.join(self.config.model_version_dir, f"scaler_churn_version_{self.datetime_suffix}.pkl")
        
        try:
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            jb.dump(scaler, scaler_path)

            return X_train_scaled, X_test_scaled, scaler_path

        except Exception as e:
            logger.error(f"Error in preparing scaler: {e}")
            raise e
    
    def full_model(self, X_train, X_test):
        """Create the base model and scaler."""
        logger.info("Creating base model and scaler")
        
        model = self.get_base_model()
        X_train_scaled, X_test_scaled, scaler_path= self.scaler(X_train, X_test)
        
        base_model_path = os.path.join(self.config.model_version_dir, f"base_model_churn_{self.datetime_suffix}.pkl")
        jb.dump(model, base_model_path)
        logger.info(f"Base model saved: {base_model_path}")
        # mlflow.log_artifact(str(scaler_path))
        # mlflow.log_artifact(str(base_model_path))
        return model, base_model_path, scaler_path, X_train_scaled, X_test_scaled
        