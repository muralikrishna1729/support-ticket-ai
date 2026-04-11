import json
import time
from aws.sqs_client import receive_messages, delete_message
from src.services.ticket_service import process_ticket_background
from src.logger import logger


def run_worker():
    logger.info("=" * 50)
    logger.info("  SmartTicket Worker Started — Polling SQS...")
    logger.info("=" * 50)
    while True:
        try:
            messages = receive_messages(max_messages=1)
            if not messages:
                time.sleep(2)
                continue
            for message in messages:
                # 1. Parse the string inside 'Body' into a dictionary
                body = json.loads(message["Body"])
                
                # 2. Get the ticket_id from the 'body' dict, NOT the 'message' dict
                # Use .get() to avoid KeyError if the key is missing
                ticket_id = body.get("ticket_id") 

                if ticket_id is None:
                    logger.warning("Message received without ticket_id. Deleting.")
                    delete_message(message["ReceiptHandle"])
                    continue

                logger.info(f"Worker picked up ticket_id: {ticket_id}")

                process_ticket_background(ticket_id)
                delete_message(message["ReceiptHandle"])
        except KeyboardInterrupt:
            logger.info("Worker stopped by user.")
            break
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            time.sleep(2)   # wait before retrying

if __name__ == "__main__":
    run_worker()