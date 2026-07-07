import sys
from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def run(self, source_path: str):
        logger.info("========== TRAINING PIPELINE STARTED ==========")
        try:
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion(source_path)
            transformation = DataTransformation()
            (
                X_train, X_test,
                y_train_cat,  y_test_cat,
                y_train_type, y_test_type,
                class_weight_dict,
                le_category, le_issue_type
            ) = transformation.initiate_data_transformation(train_path, test_path)

            trainer = ModelTrainer()
            f1_cat, f1_type = trainer.initiate_model_trainer(
                X_train, X_test,
                y_train_cat,  y_test_cat,
                y_train_type, y_test_type,
                class_weight_dict
            )
            logger.info(f"Category F1: {f1_cat} | Issue Type F1: {f1_type}")
            logger.info("========== TRAINING PIPELINE COMPLETED ==========")
            return f1_cat, f1_type
        
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    SOURCE   = "notebook/data/dataset-tickets-multi-lang-4-20k.csv"
    pipeline = TrainPipeline()
    f1_cat, f1_type = pipeline.run(SOURCE)
    print(f"\n✅ Training complete!")
    print(f"   Category F1   : {f1_cat}")
    print(f"   Issue Type F1 : {f1_type}")