from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class TicketCreate(BaseModel):
    ticket:str

class TicketResponse(BaseModel):
    id: int
    ticket_text: str
    category: Optional[str] = None
    issue_type: Optional[str] = None
    auto_response: Optional[str] = None
    status: str
    confidence: Optional[float]
    needs_review: bool
    created_at: datetime
    updated_at: datetime

