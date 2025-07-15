from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import mlflow
from dotenv import load_dotenv
from controller.prediction import router as prediction_router
from controller.training import router as training_router
import os
import dagshub

# Load environment variables
load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD')

# Initialize DagsHub integration
dagshub.init(repo_owner='minh120103', repo_name='E2E_ML', mlflow=True)

# Configure MLflow tracking URI
mlflow.set_tracking_uri('https://dagshub.com/minh120103/E2E_ML.mlflow')

# Set experiment
mlflow.set_experiment('Fraud_Detection_Experiment')

# Create FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for customer churn prediction and model retraining",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(prediction_router)
app.include_router(training_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Churn Prediction API. Use /docs to view the API documentation."}

if __name__ == "__main__":
    # uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True) #AWS