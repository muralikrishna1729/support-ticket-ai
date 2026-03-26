import os 
import sys
import re
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.utils.class_weight import compute_class_weight
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    tfidf_path         : str = os.path.join("models", "tfidf_vectorizer.pkl")
    le_category_path   : str = os.path.join("models", "le_category.pkl")
    le_issue_type_path : str = os.path.join("models", "le_issue_type.pkl")

class DataTransformation:
    def __init__(self):
        self.config= DataTransformationConfig()
        custom_sw = {
            'data', 'support', 'issue', 'issues', 'information',
            'provide', 'request', 'assistance', 'customer',
            'appreciate', 'regards', 'thanks', 'thank', 'dear',
            'team', 'help', 'looking', 'forward', 'greatly',
            'soon', 'problem', 'great', 'problems'
        }
        self.stop_words = ENGLISH_STOP_WORDS.union(custom_sw)

    def clean_text(self,text:str)->str:
        text = str(text).lower()
        text  = re.sub(r"[^a-zA-Z]", " ", text)
        words = text.split()
        words = [w for w in words if w not in self.stop_words and len(w) > 2]
        return " ".join(words)

    def initiate_data_transformation(self, train_path:str,test_path:str):
        logger.info("=== Data Transformation Started ===")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            # ── Clean text ────────────────────────────────────────
            logger.info("Cleaning text...")
            train_df["clean_text"] = train_df["text"].apply(self.clean_text)
            test_df["clean_text"] = test_df["text"].apply(self.clean_text)
            # ── Drop empty rows after cleaning ────────────────────
            train_df = train_df[
                train_df['clean_text'].str.split().str.len() > 2
            ].reset_index(drop=True)
            test_df = test_df[
                test_df['clean_text'].str.split().str.len() > 2
            ].reset_index(drop=True)

            # ── Encode labels ─────────────────────────────────────
            logger.info("Encoding labels...")
            le_category   = LabelEncoder()
            le_issue_type = LabelEncoder()

            train_df['category_enc'] = le_category.fit_transform(train_df["category"])
            train_df['issue_type_enc'] = le_issue_type.fit_transform(train_df["issue_type"])

            test_df['category_enc']   = le_category.transform(test_df['category'])
            test_df['issue_type_enc'] = le_issue_type.transform(test_df['issue_type'])

            weights = compute_class_weight(
                'balanced',
                classes = np.unique(train_df['category_enc']),
                y = train_df['category_enc']
            )   
            class_weight_dict = dict(
                zip(np.unique(train_df["category_enc"]),weights)
            ) 

             # ── TF-IDF (fit on train only!) ───────────────────────

            logger.info("Fitting TF-IDF...") 
            tfidf = TfidfVectorizer(
                max_features = 20000,
                ngram_range  = (1, 3),
                min_df       = 2,
                max_df       = 0.85,
                sublinear_tf = True,
            )
            X_train = tfidf.fit_transform(train_df["clean_text"])
            X_test  = tfidf.transform(test_df['clean_text'])

            y_train_cat  = train_df['category_enc'].values
            y_test_cat   = test_df['category_enc'].values
            y_train_type = train_df['issue_type_enc'].values
            y_test_type  = test_df['issue_type_enc'].values

            logger.info(f"X_train shape: {X_train.shape}")
            logger.info(f"X_test shape : {X_test.shape}")

            save_object(self.config.tfidf_path, tfidf)
            save_object(self.config.le_category_path,   le_category)
            save_object(self.config.le_issue_type_path, le_issue_type)

            logger.info("=== Data Transformation Completed ===")
            return (
                X_train, X_test,
                y_train_cat,  y_test_cat,
                y_train_type, y_test_type,
                class_weight_dict,
                le_category, le_issue_type
            )
        except Exception as e:
            raise CustomException(e,sys)

