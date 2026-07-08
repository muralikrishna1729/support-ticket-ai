from fastapi import APIRouter, Depends, HTTPException,BackgroundTasks
from sqlalchemy.orm import Session
from src.db import schemas
from src.db.database import sessionLocal
from typing import Optional
from src.services import ticket_service

router = APIRouter()
def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/tickets", response_model=schemas.TicketResponse)
def create_ticket(data: schemas.TicketCreate, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    ticket = ticket_service.create_ticket(db, data.ticket)
    background_tasks.add_task(ticket_service.process_ticket_background, ticket.id)
    return ticket

@router.get("/tickets/search", response_model=list[schemas.TicketResponse])
def search_tickets(
    keyword: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    return ticket_service.search_tickets(db, keyword=keyword, category=category, status=status)

@router.get("/tickets/{ticket_id}", response_model=schemas.TicketResponse)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    ticket = ticket_service.get_ticket(db, ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket

@router.get("/tickets", response_model=list[schemas.TicketResponse])
def get_all_tickets(db: Session = Depends(get_db)):
    return ticket_service.get_all_tickets(db)

@router.get("/analytics/summary")
def analytics(db:Session = Depends(get_db)):
    return ticket_service.get_analytics(db)