# Fraud Detection ML End-to-End Project
A comprehensive machine learning project for fraud detection with a complete MLOps pipeline, featuring automated training, evaluation, and prediction capabilities through a FastAPI web service.

## 🏗️ Architecture Overview
This project implements a full end-to-end ML pipeline with the following components:

- **Data Ingestion & Processing**: Automated data preparation and feature engineering
- **Model Training**: XGBoost-based fraud detection with hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics and visualization
- **MLflow Integration**: Experiment tracking and model versioning
- **FastAPI Service**: RESTful API for predictions and retraining
- **Cloud Storage**: S3 integration for model and artifact storage
- **DVC**: Data version control and pipeline management

## 📁 Project Structure
```
fraud-detection-ml/
├── src/
│   ├── fraud_detection/
│   │   ├── components/
│   │   │   ├── data_ingestion.py
│   │   │   ├── data_preparation.py
│   │   │   ├── model_trainer.py
│   │   │   └── model_evaluation.py
│   │   ├── pipeline/
│   │   │   ├── training_pipeline.py
│   │   │   └── prediction_pipeline.py
│   │   ├── entity/
│   │   │   └── config_entity.py
│   │   ├── config/
│   │   │   └── configuration.py
│   │   └── utils/
│   │       └── common.py
├── app.py
├── main.py
├── requirements.txt
├── config.yaml
├── dvc.yaml
├── params.yaml
├── .env
├── logs/
├── artifacts/
└── README.md
```

## 🚀 Features

### Machine Learning Pipeline
- **Data Processing**: Automated data ingestion with support for CSV/Excel files
- **Feature Engineering**: Robust scaling and preprocessing
- **Model Training**: XGBoost classifier with automated hyperparameter tuning
- **Model Evaluation**: Comprehensive metrics including:
  - Precision, Recall, F1-Score
  - ROC-AUC and Precision-Recall curves
  - Confusion Matrix
  - Matthews Correlation Coefficient

### MLOps Integration
- **MLflow Tracking**: Experiment tracking with DagsHub integration
- **Model Versioning**: Automated model registration and versioning
- **Artifact Management**: Automatic logging of models, scalers, and visualizations
- **Cloud Storage**: S3 integration for model and prediction storage

### API Services
- **Training Endpoint**: `/workflow/train` - Trigger model retraining
- **Prediction Endpoint**: `/fraud/` - Real-time fraud predictions
- **Health Check**: `/` - API status endpoint

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd fraud-detection-ml
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file with the following:
```env
MLFLOW_TRACKING_URI=your-tracking-url
MLFLOW_TRACKING_USERNAME=your-username
MLFLOW_TRACKING_PASSWORD=your-token
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_DEFAULT_REGION=your-aws-service-region
```

4. **Configure the project**
Update `config.yaml` with your specific settings.

**Ensure that the installed MLflow version is compatible with your Python version.**
## 📊 Usage

### Starting the API Server
```bash
python app.py
```
The API will be available at `http://localhost:8888` with interactive documentation at `http://localhost:8888/docs`.

### Training a Model

**Via API:**
```bash
curl -X POST "http://localhost:8888/workflow/train"
```

**Via Python:**
```python
from src.fraud_detection.pipeline.training_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.run_pipeline()
```

### Making Predictions

**Via API:**
```bash
curl -X POST "http://localhost:8888/fraud/" \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 0.5,
    "feature2": 1.2,
    "feature3": -0.3
  }'
```

**Response:**
```json
{
  "prediction": "fraud",
  "probability": 0.85,
  "confidence": "high"
}
```

### Running the Complete Pipeline
```bash
python main.py
```

This will execute:
1. Data ingestion and preprocessing
2. Model preparation and training
3. Model evaluation and metrics calculation
4. Cloud storage upload
5. Cleanup of temporary files

## 🔧 Configuration

The project uses a centralized configuration system through `ConfigurationManager`:

- **Data Ingestion**: File paths and data processing settings
- **Model Training**: XGBoost parameters and training configuration
- **Evaluation**: Metrics and visualization settings
- **Cloud Storage**: S3 bucket and upload configurations
- **MLflow**: Experiment tracking and model registry settings

## 📈 Model Performance
The model includes comprehensive evaluation metrics:

- **Classification Metrics**: Precision, Recall, F1-Score
- **Probability Metrics**: ROC-AUC, Average Precision
- **Visual Analytics**: Confusion Matrix, ROC Curve, Precision-Recall Curve
- **Confidence Scoring**: Average prediction confidence with threshold alerts

## 🔄 MLOps Workflow
- **Data Ingestion**: Automated data loading and validation
- **Model Training**: XGBoost training with hyperparameter optimization
- **Evaluation**: Comprehensive model assessment
- **Registration**: Automatic model registration in MLflow
- **Deployment**: Model serving through FastAPI
- **Monitoring**: Prediction confidence monitoring and alerts

## 📦 Key Components
### Training Pipeline
`WorkflowRunner` orchestrates the complete training workflow:

- Data preparation through `DataPreparationPipeline`
- Model preparation via `ModelPreparationPipeline`
- Training and evaluation with `TrainEvaluationPipeline`
- Cloud storage upload using `CloudStoragePushPipeline`

### Prediction Pipeline
`ChurnController` handles prediction requests:

- Dynamic model loading based on version/run_id
- Batch prediction processing
- Result storage and S3 upload
- Confidence monitoring and alerts

### Model Training
`TrainAndEvaluateModel` provides:

- XGBoost model training
- Hyperparameter tuning with `RandomizedSearchCV`
- Comprehensive evaluation metrics
- Visualization generation
- MLflow integration

## 🚨 Monitoring & Alerts
The system includes built-in monitoring:

- **Confidence Thresholds**: Alerts when prediction confidence falls below 70%
- **Model Performance**: Automatic evaluation metrics tracking
- **Data Drift**: Support for data quality monitoring
- **MLflow Integration**: Complete experiment and model lifecycle tracking

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links
- **MLflow UI**: Access your MLflow tracking server for experiment monitoring
- **API Documentation**: http://localhost:8888/docs when running locally
- **DagsHub**: Integration for collaborative ML development

## 📞 Support
For issues and questions:

- Check the logs in the logs directory
- Review the MLflow experiments for training issues
- Verify configuration in config.yaml
- Check the FastAPI documentation at /docs