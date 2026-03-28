## this includes the db operation for particular functions when calling routes

from sqlalchemy.orm import Session
from src.db.database import sessionLocal
from src.db import models
from src.logger import logger
from src.pipeline.predict_pipeline import PredictPipeline


pipeline = PredictPipeline()
def create_ticket(db:Session,ticket_text:str):
    ticket = models.Ticket(
        ticket_text = ticket_text,
        status = "pending"
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    logger.info(f"Ticket created → ID: {ticket.id}")

    return ticket

def process_ticket_background(ticket_id:int):
    db = sessionLocal()
    try:
        ticket = db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
        if not ticket:
            logger.error(f"Ticket {ticket_id} not found in background")
            return 
        ticket.status = "processing"
        db.commit()
        logger.info(f"Ticket {ticket_id} → processing")
        result = pipeline.predict(ticket.ticket_text)
        ticket.category      = result["category"]
        ticket.issue_type    = result["issue_type"]
        ticket.auto_response = result["auto_response"]
        ticket.status        = "completed"
        db.commit()
        logger.info(f"Ticket {ticket_id} → completed | {result['category']}")
    except Exception as e:
        logger.error(f"Ticket {ticket_id} failed: {str(e)}")
        ticket.status = "failed"
        db.commit()

    finally:
        db.close()


def get_ticket(db:Session,ticket_id:int):
    return db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
    