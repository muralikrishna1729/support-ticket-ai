import os 
import sys 
import json 
import joblib 
import numpy as np 
from src.exception import CustomException
from src.logger import logger
from sklearn.metrics import f1_score



def save_object(file_path:str,obj)->None:
    """ Save the model and use it for further..."""

    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        joblib.dump(obj,file_path)
        logger.info("Object saved → {file_path}")
    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(clf, X_val,y_val)->float:
    try:
        y_pred = clf.predict(X_val)
        score = f1_score(y_val,y_pred, average = 'weighted')
        return round(float(score),4)
    
    except Exception as e:
        raise CustomException(e,sys)

def save_json(file_path:str,data:dict)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"w") as f:
            json.dump(data,f, indent=2)
        logger.info(f"JSON saved → {file_path}")
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path:str)->dict:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model Was not Found:{file_path}")
        obj = joblib.load(file_path)
        logger.info(f"Object loaded ← {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e,sys)