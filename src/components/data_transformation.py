import sys
from dataclasses import dataclass
import pandas as pd 
from sklearn.pipeline import pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocesser_obj_file_path = os.path.join("artifacts",'preprocesser.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def data_transformation_object(self):
        try:

        except:
              
