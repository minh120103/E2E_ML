import os
import yaml
import boto3
import httpx
import asyncio
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load .env and YAML config
load_dotenv()
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load env vars
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")

app = FastAPI()

# Upload file to S3
def upload_to_s3(local_file_path: str, bucket_name: str, object_key: str):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    s3.upload_file(local_file_path, bucket_name, object_key)

# Notify webhook
async def notify_webhook(status: str, filename: str):
    if not WEBHOOK_URL:
        print("No webhook URL provided.")
        return
    async with httpx.AsyncClient() as client:
        await client.post(WEBHOOK_URL, json={"status": status, "file": filename})

# Background process
def process_and_upload(file_path: str, bucket_name: str, object_key: str):
    try:
        upload_to_s3(file_path, bucket_name, object_key)
        asyncio.run(notify_webhook("success", os.path.basename(file_path)))
    except Exception:
        asyncio.run(notify_webhook("failed", os.path.basename(file_path)))
        raise

# Upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    save_path = config['data_ingestion']['local_data_file']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(await file.read())

    bucket = config['data_ingestion']['bucket_name']
    object_key = f"data/{os.path.basename(save_path)}"

    background_tasks.add_task(process_and_upload, save_path, bucket, object_key)

    return JSONResponse({"message": "File saved and upload started", "local_path": save_path})
