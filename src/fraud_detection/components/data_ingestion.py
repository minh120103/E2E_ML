import os 
from src.fraud_detection import logger
from src.fraud_detection.utils.common import get_size
from src.fraud_detection.entity.config_entity import DataIngestionConfig
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pathlib import Path
from datetime import datetime


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.datetime_suffix = datetime.now().strftime('%Y%m%dT%H%M%S')


    def load_data(self):
        """Load the dataset from CSV file and save versioned copy to cloud storage."""
        try:
            logger.info(f"Loading data from {self.config.local_data_file}")
            df = pd.read_csv(self.config.local_data_file)

            logger.info(f"Loaded dataset with {len(df)} rows")
            logger.info(f"Columns found: {list(df.columns)}")

            return df

        except Exception as e:
            logger.error(f"Error while loading data: {e}")
            raise

    def save_data(self, df, df_processed):
            input_data_versioned_name = f"input_raw_data_version_{self.datetime_suffix}.csv"
            processed_data_versioned_name = f"processed_data_version_{self.datetime_suffix}.csv"
            input_data_versioned_path = Path(self.config.data_version_dir) / input_data_versioned_name
            processed_data_versioned_name = Path(self.config.data_version_dir) / processed_data_versioned_name
            if not input_data_versioned_path.exists():
                df.to_csv(input_data_versioned_path, index=False)
            if not processed_data_versioned_name.exists():
                df_processed.to_csv(processed_data_versioned_name, index=False)
                logger.info(f"Created versioned input data file: {input_data_versioned_path}")
                logger.info(f"Created versioned processed data file: {processed_data_versioned_name}")
            else:
                logger.info(f"Versioned file already exists: {input_data_versioned_path}, skipping save.")
                logger.info("Continuing with local processing...")

    def data_ingestion_pipeline(self):
        """Main method to perform data ingestion."""
        logger.info("Initiating data ingestion")
        df = self.load_data()
        X, y, df_processed = self.preprocess_data(df)
        self.save_data(df, df_processed)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        logger.info("Data ingestion completed successfully")
        logger.info(f"Data shape - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, df_processed, df
    
    def split_data(self, X_res, y_res):
        """Split data into training and testing sets."""
        logger.info("Splitting data into train and test sets")
        X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=self.config.test_size,
            random_state=self.config.random_state)
        logger.info(f"Train data: {X_train.shape}, Test data: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, df_clean):
        self.rows_processed = 0
        logger.info(f"Starting preprocessing of {len(df_clean)} rows...")
        logger.info(f"Initial columns: {list(df_clean.columns)}")
        
        df_processed = self.process_data_for_churn(df_clean)
        logger.info(f"After feature engineering columns: {list(df_processed.columns)}")
        logger.info(f"Data shape after feature engineering: {df_processed.shape}")
        logger.info(f"First few rows of feature engineered data: \n{df_processed.head()}")
        
        logger.info(f"After encoding columns: {list(df_processed.columns)}")
        logger.info(f"Data shape after encoding: {df_processed.shape}")
        
        df_processed = df_processed.dropna()
        logger.info(f"Data shape after removing NaN: {df_processed.shape}")

        if "isFraud" not in df_processed.columns:
            logger.error(f"isFraud column not found! Available columns: {list(df_processed.columns)}")
            raise KeyError("isFraud column is missing after preprocessing")

        X = df_processed.drop("isFraud", axis=1)
        y = df_processed["isFraud"]

        logger.info(f"X shape (features): {X.shape}")
        logger.info(f"y shape (target): {y.shape}")
        
        logger.info(f"Target variable shape: {y.shape}")
        logger.info(f"NaN count in target: {y.isna().sum()}")
        logger.info(f"Target variable distribution: \n{y.value_counts(normalize=True)}")
        
        if y.isna().sum() > 0:
            logger.warning(f"Found {y.isna().sum()} NaN values in target variable, filling with 0")
            y = y.fillna(0)
        
        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            logger.warning(f"Found NaN values in feature columns: {nan_cols}")
            X = X.fillna(0)
        
        logger.info(f"Final feature matrix shape: {X.shape}")
        logger.info(f"Final target vector shape: {y.shape}")
        
        class_distribution = y.value_counts(normalize=True)
        logger.info(f"Target variable distribution (normalized): \n{class_distribution}")

        imbalance_threshold = 0.4

        if class_distribution.min() < imbalance_threshold:
            logger.info("Target variable is imbalanced. Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"Resampled feature matrix shape: {X_res.shape}")
            logger.info(f"Resampled target distribution: \n{y_res.value_counts()}")
        else:
            logger.info("Target variable is balanced. Skipping SMOTE.")
            X_res, y_res = X, y
        return X_res, y_res, df_processed

    def process_data_for_churn(self,df_input):
        df_input['MoneySpentSender'] = df_input['oldbalanceOrg'] - df_input['newbalanceOrig']
        df_input['MoneyReceiver'] = df_input['oldbalanceDest'] - df_input['newbalanceDest']
        df_input = df_input.drop(columns=['newbalanceOrig'])
        df_input = df_input.drop(columns=['newbalanceDest'])
        df_input = df_input.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])
        df_input = pd.get_dummies(df_input, columns=['type'], dtype=int)

        return df_input