"""
Worker — polls SQS and processes tickets.
Run: python worker.py
"""
import json
import time
from aws.sqs_client import receive_messages, delete_message
from src.services.ticket_service import process_ticket_background
from src.logger import logger


def run_worker():
    logger.info("=" * 50)
    logger.info("  SmartTicket Worker Started")
    logger.info("  Polling SQS...")
    logger.info("=" * 50)

    while True:
        try:
            messages = receive_messages(max_messages=1)

            if not messages:
                time.sleep(2)
                continue

            for message in messages:
                body      = json.loads(message["Body"])
                ticket_id = body["ticket_id"]
                logger.info(f"Worker picked ticket_id: {ticket_id}")
                process_ticket_background(ticket_id)
                delete_message(message["ReceiptHandle"])

        except KeyboardInterrupt:
            logger.info("Worker stopped.")
            break
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            time.sleep(2)


if __name__ == "__main__":
    run_worker()