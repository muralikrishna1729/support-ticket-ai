import sys
from dataclasses import dataclass
import pandas as pd 
from sklearn.pipeline import pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging


class DataTransformationConfig:
    