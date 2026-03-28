import sys
from src.exception import CustomException
from src.logger import logger
from src.pipeline.predict_pipeline import PredictPipeline
from fastapi import FastAPI
from pydantic import BaseModel
from src.api.routes import router
from src.db.database import Base, engine

app = FastAPI(title = "SmartTicket AI")
Base.metadata.create_all(bind = engine)
app.include_router(router)
pipeline = PredictPipeline()

class InputITicket(BaseModel):
    ticket:str

@app.get("/")
def home():
    return {"message":"SmartTicket is Live"}

@app.get("/health")
def health():
    return {"status": "healthy"}





