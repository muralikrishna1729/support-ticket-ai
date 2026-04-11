"""
Run once to upload all model artifacts to S3.
Usage: python scripts/upload_models.py
"""
import sys 
import os 

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from aws.s3_client import upload_file
from src.logger import logger

MODEL_FILES = [
    "models/clf_category.pkl",
    "models/clf_issue_type.pkl",
    "models/tfidf_vectorizer.pkl",
    "models/le_category.pkl",
    "models/le_issue_type.pkl",
    "models/model_scores.json",
]

def upload_all_models():
    logger.info("Uploading models to S3...")
    success = 0
    for local_path in MODEL_FILES:
        if not os.path.exists(local_path):
            logger.warning(f"File not found: {local_path}")
            continue
        s3_key = local_path
        if upload_file(local_path,s3_key):
            success+=1

    logger.info(f"Uploaded {success}/{len(MODEL_FILES)} files")
    print(f"\n Done! {success} files uploaded to S3.")

if __name__ == "__main__":
    upload_all_models()
