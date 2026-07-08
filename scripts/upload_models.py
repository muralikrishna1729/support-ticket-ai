# scripts/upload_models.py
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

def upload_all():
    logger.info("Uploading models to S3...")
    success = 0
    for local_path in MODEL_FILES:
        if not os.path.exists(local_path):
            logger.warning(f"Not found: {local_path}")
            continue
        if upload_file(local_path, s3_key=local_path):
            success += 1
    print(f"\n✅ {success}/{len(MODEL_FILES)} files uploaded")

if __name__ == "__main__":
    upload_all()