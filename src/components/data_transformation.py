import os
import sys
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.utils.class_weight import compute_class_weight
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    tfidf_path         : str = os.path.join("models", "tfidf_vectorizer.pkl")
    le_category_path   : str = os.path.join("models", "le_category.pkl")
    le_issue_type_path : str = os.path.join("models", "le_issue_type.pkl")

# Negation tagging — keep identical to benchmark_techniques.py so training
# and inference see the same transformation.
NEG_CUES = {
    "not", "no", "nor", "never", "neither", "nothing", "nobody", "none",
    "cannot", "unable", "without", "lack", "lacks", "lacking", "missing",
    "fail", "fails", "failed", "failing", "deny", "denied",
    "refuse", "refused", "broken", "outage",
}
NEG_WINDOW = 3

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        custom_sw = {
            'data', 'support', 'issue', 'issues', 'information',
            'provide', 'request', 'assistance', 'customer',
            'appreciate', 'regards', 'thanks', 'thank', 'dear',
            'team', 'help', 'looking', 'forward', 'greatly',
            'soon', 'problem', 'great', 'problems'
        }
        self.stop_words = ENGLISH_STOP_WORDS.union(custom_sw)

    def _expand_contractions(self, text: str) -> str:
        """Turn n't contractions into standalone 'not' BEFORE punctuation
        stripping, so the negation cue survives cleaning."""
        text = re.sub(r"n't\b", " not", text)          # don't / doesn't / isn't ...
        text = re.sub(r"\bcannot\b", "can not", text)  # cannot -> can not
        return text

    def clean_text(self, text: str) -> str:
        """Aggressive cleaning with negation tagging (v3 recipe).
        Cue words are kept and the next NEG_WINDOW content words get a
        NEG_ prefix so TF-IDF can separate polarity:
        "can access"      -> "access"
        "cannot access"   -> "not NEG_access"
        """
        text = str(text).lower()
        text = self._expand_contractions(text)
        text = re.sub(r"[^a-z]", " ", text)
        out, neg_left = [], 0
        for w in text.split():
            if w in NEG_CUES:
                out.append(w)
                neg_left = NEG_WINDOW
                continue
            if w in self.stop_words or len(w) <= 2:
                continue
            if neg_left > 0:
                out.append("NEG_" + w)
                neg_left -= 1
            else:
                out.append(w)
        return " ".join(out)


    def initiate_data_transformation(self, train_path: str, test_path: str):
        logger.info("=== Data Transformation (Negation-tagged TF-IDF) Started ===")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            # Clean text
            logger.info("Cleaning text (negation tagging enabled)...")
            train_df['clean_text'] = train_df['text'].apply(self.clean_text)
            test_df['clean_text']  = test_df['text'].apply(self.clean_text)

            # Drop rows that became too short after cleaning
            train_df = train_df[train_df['clean_text'].str.split().str.len() > 2].reset_index(drop=True)
            test_df  = test_df[test_df['clean_text'].str.split().str.len() > 2].reset_index(drop=True)

            # Encode labels
            logger.info("Encoding labels...")
            le_category   = LabelEncoder()
            le_issue_type = LabelEncoder()

            train_df['category_enc']   = le_category.fit_transform(train_df['category'])
            train_df['issue_type_enc'] = le_issue_type.fit_transform(train_df['issue_type'])
            test_df['category_enc']    = le_category.transform(test_df['category'])
            test_df['issue_type_enc']  = le_issue_type.transform(test_df['issue_type'])

            # Class weights
            weights = compute_class_weight(
                'balanced',
                classes = np.unique(train_df['category_enc']),
                y       = train_df['category_enc']
            )
            class_weight_dict = dict(
                zip(np.unique(train_df['category_enc']), weights)
            )
            logger.info("Class weights computed OK")

            # Vectorize (same params as the v1 TF-IDF pipeline)
            logger.info("Fitting TF-IDF vectorizer...")
            tfidf = TfidfVectorizer(
                max_features=20000, ngram_range=(1, 3),
                min_df=2, max_df=0.85, sublinear_tf=True,
            )
            X_train = tfidf.fit_transform(train_df['clean_text'])
            X_test  = tfidf.transform(test_df['clean_text'])
            logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

            y_train_cat  = train_df['category_enc'].values
            y_test_cat   = test_df['category_enc'].values
            y_train_type = train_df['issue_type_enc'].values
            y_test_type  = test_df['issue_type_enc'].values

            # Save artifacts
            save_object(self.data_transformation_config.tfidf_path,         tfidf)
            save_object(self.data_transformation_config.le_category_path,   le_category)
            save_object(self.data_transformation_config.le_issue_type_path, le_issue_type)

            logger.info("=== Data Transformation Completed ===")

            return (
                X_train, X_test,
                y_train_cat,  y_test_cat,
                y_train_type, y_test_type,
                class_weight_dict,
                le_category, le_issue_type
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation()
