## this includes the db operation for particular functions when calling routes

from sqlalchemy.orm import Session
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
    return ticket

def process_ticket(db:Session,ticket_id:int):
    ticket = db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
    if not ticket:
        return None 
    ticket.status = "processing"
    db.commit()
    try:
        result = pipeline.predict(ticket.ticket_text)
        ticket.category = result["category"]
        ticket.issue_type = result["issue_type"]
        ticket.auto_response = result["auto_response"]
        ticket.status = "completed"
    
    except Exception as e:
        ticket.status = "Failed"
    db.commit()
    db.refresh(ticket)
    return ticket

def get_ticket(db:Session,ticket_id:int):
    return db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
    