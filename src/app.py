from fastapi import FastAPI
from src.api.routes import router
from src.db.database import Base, engine
from src.logger import logger

Base.metadata.create_all(bind = engine)
logger.info("Database tables created ✅")

app = FastAPI(
    title       = "SmartTicket AI",
    description = "ML-powered customer support ticket classifier",
    version     = "1.0.0"
)
app.include_router(router)

@app.get("/")
def home():
    return {
        "message" : "SmartTicket AI is live 🚀",
        "version" : "1.0.0",
        "docs"    : "/docs"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}