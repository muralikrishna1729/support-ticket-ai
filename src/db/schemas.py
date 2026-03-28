from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict

class TicketCreate(BaseModel):
    ticket: str

class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ticket_text: str
    category: Optional[str] = None
    issue_type: Optional[str] = None
    auto_response: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime   