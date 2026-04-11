import boto3
import json
import os
from dotenv import load_dotenv
from src.logger import logger

load_dotenv()

sqs = boto3.client(
    "sqs",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name           = os.getenv("AWS_REGION")
)
QUEUE_URL = os.getenv("SQS_QUEUE_URL")

def send_message(ticket_id:int)->bool:
    try:
        response = sqs.send_message(
            QueueUrl = QUEUE_URL,
            MessageBody = json.dumps({"ticket_id":ticket_id})
        )        
        logger.info(f"SQS message sent → ticket_id: {ticket_id} | MessageId: {response['MessageId']}")
        return True 
    except Exception as e:
        logger.error(f"SQS send failed: {str(e)}")
        return False

def receive_messages(max_messages:int=1):
    try:
        response = sqs.receive_message(
            QueueUrl = QUEUE_URL,
            MaxNumberOfMessages = max_messages,
            WaitTimeSeconds = 5 
        )
        return response.get("Messages", [])
    except Exception as e:
        logger.error(f"SQS receive failed: {str(e)}")
        return []

def delete_message(receipt_handle: str) -> None:
    """Delete processed message from queue."""
    try:
        sqs.delete_message(
            QueueUrl      = QUEUE_URL,
            ReceiptHandle = receipt_handle
        )
        logger.info("SQS message deleted after processing")
    except Exception as e:
        logger.error(f"SQS delete failed: {str(e)}")