from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime 
from .database import Base


class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(Integer, primary_key=True, index=True)
    ticket_text = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    issue_type = Column(String, nullable=True)
    auto_response = Column(Text, nullable=True)
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)