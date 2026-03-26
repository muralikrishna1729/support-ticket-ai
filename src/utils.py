import os 
import sys 
import json 
import dlib 
import joblib 
import numpy as np 
from src.exception import CustomException
from src.logger import logger


def save_object(file_path:str,obj)->None:
    """ Save the model and use it for further..."""

    try:
        pass
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        joblib.dump(obj,file_path)
        logger.info("Object saved → {file_path}")
    except Exception as e:
        raise CustomException(e,sys)


