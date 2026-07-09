import os
import sys
from src.logger import logger
from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformation

MODEL_PATHS = {
    "tfidf"         : "models/tfidf_vectorizer.pkl",
    "clf_category"  : "models/clf_category.pkl",
    "clf_issue_type": "models/clf_issue_type.pkl",
    "le_category"   : "models/le_category.pkl",
    "le_issue_type" : "models/le_issue_type.pkl",
}

RESPONSES = {
    "Technical Support"              : "Our technical team will respond within 4 hours.",
    "Billing and Payments"           : "Our billing team will contact you within 24 hours.",
    "IT Support"                     : "IT support ticket created. Resolution: 2-4 hours.",
    "Customer Service"               : "A representative will reach out shortly.",
    "Product Support"                : "Our product team will assist within 8 hours.",
    "Returns and Exchanges"          : "Your return request has been initiated.",
    "Service Outages and Maintenance": "We are aware and working to resolve immediately.",
    "Sales and Pre-Sales"            : "A sales rep will contact you within 1 business day.",
    "Human Resources"                : "HR will respond within 2 business days.",
    "General Inquiry"                : "We will respond within 24 hours.",
}

def ensure_models_exist():
    """Download from S3 if models missing locally."""
    missing = [name for name, path in MODEL_PATHS.items()
               if not os.path.exists(path)]
    if not missing:
        logger.info("All models found locally ✅")
        return

    logger.info(f"{len(missing)} model(s) missing → downloading from S3")

    from aws.s3_client import download_file

    failed = []
    for name in missing:
        local_path = MODEL_PATHS[name]
        success = download_file(s3_key=local_path, local_path=local_path)
        if not success:
            failed.append(local_path)

    if failed:
        # Log which files failed
        for f in failed:
            logger.error(f"Failed to download: {f}")
        raise RuntimeError(
            f"Could not download {len(failed)} model(s) from S3: {failed}"
        )

    logger.info("All models downloaded from S3 ✅")

    
class PredictPipeline:
    def __init__(self):
        if os.getenv("TESTING") == "true":
            logger.info("TESTING mode: skipping model load")
            self.tfidf = self.clf_category = self.clf_issue_type = None
            self.le_category = self.le_issue_type = None
            return
        ensure_models_exist()
        logger.info("Loading model artifacts...")
        self.tfidf          = load_object(MODEL_PATHS["tfidf"])
        self.clf_category   = load_object(MODEL_PATHS["clf_category"])
        self.clf_issue_type = load_object(MODEL_PATHS["clf_issue_type"])
        self.le_category    = load_object(MODEL_PATHS["le_category"])
        self.le_issue_type  = load_object(MODEL_PATHS["le_issue_type"])
        logger.info("All artifacts loaded ✅")
    
    def predict(self,text:str)->dict:
        try:
            dt         = DataTransformation()
            clean      = dt.clean_text(text)
            text_tfidf = self.tfidf.transform([clean])

            # Category prediction + confidence
            cat_enc    = self.clf_category.predict(text_tfidf)[0]
            category   = self.le_category.inverse_transform([cat_enc])[0]

            # Confidence score
            cat_scores  = self.clf_category.decision_function(text_tfidf)[0]
            confidence  = round(float(max(cat_scores)), 4)
            needs_review = confidence < 0.60

            # Issue type prediction
            type_enc   = self.clf_issue_type.predict(text_tfidf)[0]
            issue_type = self.le_issue_type.inverse_transform([type_enc])[0]

            # Auto response
            response   = RESPONSES.get(category, "Thank you for contacting us.")
            if issue_type == "Incident":
                response = "🚨 " + response

            result = {
                "category"      : category,
                "issue_type"    : issue_type,
                "auto_response" : response,
                "clean_text"    : clean,
                "confidence"    : confidence,
                "needs_review"  : needs_review
            }
            logger.info(f"Prediction: {result}")
            return result
        except Exception as e:
            raise CustomException(e, sys)