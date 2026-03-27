import os 
import sys
from dataclasses import dataclass
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, save_json, evaluate_model

@dataclass
class ModelTrainerConfig:
    category_model_path   : str = os.path.join("models", "clf_category.pkl")
    issue_type_model_path : str = os.path.join("models", "clf_issue_type.pkl")
    scores_path           : str = os.path.join("models", "model_scores.json")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self,X_train, X_test,
        y_train_cat,  y_test_cat,
        y_train_type, y_test_type,
        class_weight_dict
    ):
        logger.info("=== Model Training Started ===")
        try:
            logger.info("Training category classifier (LinearSVC + GridSearch)...")
            cat_pipeline = Pipeline([

                ("clf", LinearSVC(
                    class_weight = class_weight_dict,
                    max_iter = 2000,
                    random_state = 42
                ))
            ])
            grid = GridSearchCV(
                cat_pipeline,
                {'clf__C':[0.1, 0.3, 0.5, 1.0, 2.0]},
                cv      = 5,
                scoring = 'f1_weighted',
                n_jobs  = -1,
                verbose = 1
            )
            grid.fit(X_train,y_train_cat)
            best_cat_model = grid.best_estimator_
            logger.info(f"Best C for category: {grid.best_params_}")

            # ── Train Issue Type Model ────────────────────────────
            logger.info("Training issue type classifier...")
            issue_clf = LinearSVC(
                class_weight = 'balanced',
                C            = 1.0,
                max_iter     = 2000,
                random_state = 42
            )
            issue_clf.fit(X_train, y_train_type)

            f1_cat = evaluate_model(best_cat_model,X_test,y_test_cat)
            f1_type = evaluate_model(issue_clf,X_test,y_test_type)
            logger.info(f"Test F1 — Category   : {f1_cat}")
            logger.info(f"Test F1 — Issue Type : {f1_type}")

            # ── Detailed report ───────────────────────────────────

            y_pred_cat = best_cat_model.predict(X_test)
            y_pred_type = issue_clf.predict(X_test)

            logger.info("Category Report:\n" +classification_report(y_test_cat,  y_pred_cat))
            logger.info("Issue Type Report:\n" + classification_report(y_test_type, y_pred_type))

            save_object(self.config.category_model_path , best_cat_model)
            save_object(self.config.issue_type_model_path, issue_clf)
            # ── Save scores ───────────────────────────────────────
            scores = {
                "model_version" : "v1.0",
                "f1_category"   : f1_cat,
                "f1_issue_type" : f1_type,
                "best_C"        : grid.best_params_['clf__C']
            }
            save_json(self.config.scores_path,scores)
            logger.info("=== Model Training Completed ===")
            return f1_cat, f1_type

        except Exception as e:
            raise CustomException(e,sys)
