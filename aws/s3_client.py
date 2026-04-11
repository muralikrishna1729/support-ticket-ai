import boto3 
import os 
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id= os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name  = os.getenv("AWS_REGION")
)

BUCKET = os.getenv("S3_BUCKET_NAME")


def upload_file(local_path:str ,s3_key:str)->bool:
    try:
        s3.upload_file(local_path,BUCKET, s3_key)
        logger.info(f"Uploaded → s3://{BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"S3 upload failed: {str(e)}")
        return False

def download_file(s3_key:str,local_path:str)->bool:
    try:
        os.makedirs(os.path.dirname(local_path),exist_ok=True)
        s3.download_file(BUCKET,s3_key, local_path)
        logger.info(f"Downloaded ← s3://{BUCKET}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"S3 download failed: {str(e)}")
        return False

def file_exists(s3_key: str) -> bool:
    """Check if a file exists in S3."""
    try:
        s3.head_object(Bucket=BUCKET, Key=s3_key)
        return True
    except:
        return False
