import sys
from src.exception import CustomException
from src.logger import logger
from src.pipeline.predict_pipeline import PredictPipeline
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title = "SmartTicket AI")
pipeline = PredictPipeline()

class InputITicket(BaseModel):
    ticket:str

@app.get("/")
def home():
    return {"message":"SmartTicket is Live"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data:InputITicket):
    result = pipeline.predict(data.ticket)
    return {
        "category"      : result["category"],
        "issue_type"    : result["issue_type"],
        "auto_response" : result["auto_response"]
    }



