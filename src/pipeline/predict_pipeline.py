import pandas as pd 
import numpy as np 
import argparse
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.logger import logger
from src.utils import load_object
RESPONSES = {
    "Technical Support"              : "Our technical team has been notified and will respond within 4 hours.",
    "Billing and Payments"           : "Our billing team will review your case and contact you within 24 hours.",
    "IT Support"                     : "IT support ticket created. Expected resolution: 2-4 hours.",
    "Customer Service"               : "A customer service representative will reach out shortly.",
    "Product Support"                : "Our product support team will assist you within 8 hours.",
    "Returns and Exchanges"          : "Your return/exchange request has been initiated.",
    "Service Outages and Maintenance": "We are aware of the issue and working to resolve it immediately.",
    "Sales and Pre-Sales"            : "A sales representative will contact you within 1 business day.",
    "Human Resources"                : "HR team received your request and will respond within 2 business days.",
    "General Inquiry"                : "Thank you for reaching out. We will respond within 24 hours.",
}
class PredictPipeline:
    def __init__(self):
        logger.info("Loading model artifacts...")
        self.tfidf          = load_object("models/tfidf_vectorizer.pkl")
        self.clf_category   = load_object("models/clf_category.pkl")
        self.clf_issue_type = load_object("models/clf_issue_type.pkl")
        self.le_category    = load_object("models/le_category.pkl")
        self.le_issue_type  = load_object("models/le_issue_type.pkl")
        logger.info("All artifacts loaded successfully!")
    
    def predict(self,text:str)->dict:
        try:
            dt = DataTransformation()
            clean = dt.clean_text(text)
            text_tfidf =self.tfidf.transform([clean])
            cat_enc = self.clf_category.predict(text_tfidf)[0]
            category = self.le_category.inverse_transform([cat_enc])[0]
            type_enc    = self.clf_issue_type.predict(text_tfidf)[0]
            issue_type  = self.le_issue_type.inverse_transform([type_enc])[0]

            response = RESPONSES.get(category, "Thanks for Contacting us.")
            result  = {
                "category" : category,
                "issue_type": issue_type,
                "auto_response": response,
                "clean_text":text
            }
            logger.info(f"Prediction: {result}")
            return result

        except Exception as e:
            raise CustomException(e,sys)      

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticket", type=str,required=True,help = "enter your query")
    Predict = PredictPipeline() 
    args = parser.parse_args()

    result = Predict.predict(args.ticket)
    print(result)
        