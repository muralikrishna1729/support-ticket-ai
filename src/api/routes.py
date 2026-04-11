## this includes creating routes and defining their functionality using db operations.


from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from src.db.database import sessionLocal
from src.db import schemas
from src.services import ticket_service
from aws.sqs_client import send_message
from src.logger import logger

router = APIRouter()
def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/tickets", response_model = schemas.TicketResponse)
def create_ticket(data:schemas.TicketCreate,background_tasks: BackgroundTasks, db:Session = Depends(get_db)):
    ticket = ticket_service.create_ticket(db, data.ticket)
    # background_tasks.add_task(
    #     ticket_service.process_ticket_background(ticket.id),
    #     ticket.id
    # )
    sent = send_message(ticket.id)
    if not sent:
        logger.warning(f"SQS send failed for ticket {ticket.id} — falling back to sync")
        ticket = ticket_service.process_ticket_background(ticket.id)

    return ticket

@router.get("/tickets/{ticket_id}", response_model=schemas.TicketResponse)
def get_ticket(ticket_id:int,db:Session = Depends(get_db)):
    ticket = ticket_service.get_ticket(db,ticket_id)
    if not ticket:
        raise HTTPException(status_code = 404, detail = "Ticket was not found")
    return ticket
