import boto3
import os
import time
from dotenv import load_dotenv
from src.logger import logger

cloudwatch = boto3.client(
    "cloudwatch",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name           = os.getenv("AWS_REGION")
)

NAMESPACE = "SmartTicketAI"

def put_metric(metric_name: str, value: float, unit: str = "Count") -> None:
    """Send a custom metric to CloudWatch."""
    try:
        cloudwatch.put_metric_data(
            Namespace  = NAMESPACE,
            MetricData = [{
                "MetricName" : metric_name,
                "Value"      : value,
                "Unit"       : unit,
                "Dimensions" : [
                    {"Name": "Environment", "Value": "production"}
                ]
            }]
        )
        logger.info(f"CloudWatch metric → {metric_name}: {value}")
    except Exception as e:
        logger.error(f"CloudWatch metric failed: {str(e)}")

def log_prediction(category: str, confidence: float,
                   needs_review: bool, processing_time_ms: float) -> None:
    """Log prediction metrics to CloudWatch."""
    put_metric("PredictionCount",       1)
    put_metric("ConfidenceScore",       confidence,         "None")
    put_metric("ProcessingTimeMs",      processing_time_ms, "Milliseconds")
    if needs_review:
        put_metric("LowConfidenceCount", 1)

def log_error(error_type: str = "PredictionError") -> None:
    """Log errors to CloudWatch."""
    put_metric(error_type, 1)