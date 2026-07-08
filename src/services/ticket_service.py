from sqlalchemy import func
from sqlalchemy.orm import Session
from src.db import models
from src.db.database import sessionLocal
from src.logger import logger
from src.pipeline.predict_pipeline import PredictPipeline

pipeline = PredictPipeline()

def create_ticket(db: Session, ticket_text: str):
    logger.info("Creating a new ticket...")
    ticket = models.Ticket(
        ticket_text  = ticket_text,
        status = "pending"
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    logger.info(f"Ticket created with ID: {ticket.id}")
    return ticket

def process_ticket_background(ticket_id: int):
    db = sessionLocal()
    try:
        ticket = db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()
        if not ticket:
            logger.error(f"Ticket ID {ticket_id} not found.")
            return
        ticket.status = "processing"
        db.commit()
        logger.info(f"Processing ticket ID {ticket_id}...")
        result = pipeline.predict(ticket.ticket_text)
        ticket.category      = result["category"]
        ticket.issue_type    = result["issue_type"]
        ticket.auto_response = result["auto_response"]
        ticket.confidence    = result["confidence"]
        ticket.needs_review  = result["needs_review"]
        ticket.status = "completed"
        db.commit()
        logger.info(f"Ticket ID {ticket_id} processed successfully.")

    except Exception as e:
        logger.error(f"Error processing ticket ID {ticket_id}: {str(e)}")
        ticket.status = "failed"
        db.commit()
    finally:
        db.close()


def get_ticket(db: Session, ticket_id: int):
    logger.info(f"Fetching ticket ID {ticket_id}...")
    return db.query(models.Ticket).filter(models.Ticket.id == ticket_id).first()

def get_all_tickets(db: Session, limit:int= 50):
    logger.info("Fetching all tickets...")
    return db.query(models.Ticket)\
        .order_by(models.Ticket.created_at.desc())\
        .limit(limit).all()

def search_tickets(db: Session, keyword=None, category=None, status=None):
    logger.info(f"Searching tickets with query: {keyword}")
    query = db.query(models.Ticket)
    if keyword:
        query = query.filter(models.Ticket.ticket_text.ilike(f"%{keyword}%"))
    if category:
        query = query.filter(models.Ticket.category == category)
    if status:
        query = query.filter(models.Ticket.status == status)
    return query.order_by(models.Ticket.created_at.desc()).limit(50).all()

def get_analytics(db: Session):
    logger.info("Fetching analytics data...")
    total = db.query(models.Ticket).count()
    completed = db.query(models.Ticket).filter(models.Ticket.status == "completed").count()
    pending = db.query(models.Ticket).filter(models.Ticket.status == "pending").count()
    failed = db.query(models.Ticket).filter(models.Ticket.status == "failed").count()
    needs_review = db.query(models.Ticket).filter(models.Ticket.needs_review == True).count()
    avg_conf = db.query(func.avg(models.Ticket.confidence)).filter(models.Ticket.confidence != None).scalar()
    cat_counts = db.query(models.Ticket.category, func.count(models.Ticket.id).label("count")).filter(models.Ticket.category != None).group_by(models.Ticket.category).all()
    type_counts = db.query(models.Ticket.issue_type, func.count(models.Ticket.id).label('count')).filter(models.Ticket.issue_type!=None).group_by(models.Ticket.issue_type).all()

    return {
        "total": total,
        "completed": completed,
        "pending": pending,
        "failed": failed,
        "needs_review": needs_review,
        'avg_confidence': round(avg_conf, 4) if avg_conf else 0,
        "category_counts": {cat: count for cat, count in cat_counts},
        "issue_type_counts": {itype: count for itype, count in type_counts}
    }
