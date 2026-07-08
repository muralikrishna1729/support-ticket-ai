import boto3
import json
import os
from dotenv import load_dotenv
from src.logger import logger
from aws.ses_client import send_failure_alert
from src.services.ticket_service import process_ticket_background


load_dotenv()

sqs = boto3.client(
    "sqs",
    aws_access_key_id     = os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name           = os.getenv("AWS_REGION")
)

QUEUE_URL = os.getenv("SQS_QUEUE_URL")


def send_message(ticket_id: int) -> bool:
    try:
        response = sqs.send_message(
            QueueUrl    = QUEUE_URL,
            MessageBody = json.dumps({"ticket_id": ticket_id})
        )
        logger.info(f"SQS sent → ticket_id: {ticket_id} | MsgId: {response['MessageId']}")
        return True
    except Exception as e:
        logger.error(f"SQS send failed: {str(e)}")
        return False


def receive_messages(max_messages: int = 1):
    try:
        response = sqs.receive_message(
            QueueUrl            = QUEUE_URL,
            MaxNumberOfMessages = max_messages,
            WaitTimeSeconds     = 5,
        )
        return response.get("Messages", [])
    except Exception as e:
        logger.error(f"SQS receive failed: {str(e)}")
        return []


def delete_message(receipt_handle: str) -> None:
    try:
        sqs.delete_message(
            QueueUrl      = QUEUE_URL,
            ReceiptHandle = receipt_handle
        )
        logger.info("SQS message deleted ✅")
    except Exception as e:
        logger.error(f"SQS delete failed: {str(e)}")


def poll_and_process(sqs, queue_url):
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,
        WaitTimeSeconds=10
    )
    messages = response.get("Messages", [])
    for msg in messages:
        receipt_handle = msg["ReceiptHandle"]
        try:
            body = json.loads(msg["Body"])
            ticket_id = body["ticket_id"]

            process_ticket_background(ticket_id)

            # Delete only after successful processing
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)

        except Exception as e:
            logger.error(f"SQS message processing failed: {str(e)}")
            try:
                send_failure_alert(
                    ticket_id=body.get("ticket_id", "unknown") if 'body' in dir() else "unknown",
                    error_detail=str(e)
                )
            except Exception:
                pass
            # Don't delete — let it retry or go to DLQ per your redrive policy