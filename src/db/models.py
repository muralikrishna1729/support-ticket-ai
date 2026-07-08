from datetime import datetime
from sqlalchemy import String, Text, Float, Boolean, func
from sqlalchemy.orm import Mapped, mapped_column
from src.db.database import Base

class Ticket(Base):
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticket_text: Mapped[str] = mapped_column(Text, nullable=False)
    category:      Mapped[str | None] = mapped_column(String(100), index=True)
    issue_type:    Mapped[str | None] = mapped_column(String(100), index=True)
    auto_response: Mapped[str | None] = mapped_column(Text)
    status:        Mapped[str]       = mapped_column(String(20), default="pending", index=True)
    confidence:    Mapped[float | None] = mapped_column(Float)
    needs_review:  Mapped[bool]      = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())