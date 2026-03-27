import sys 
import os 
import argparse
from src.exception import CustomException
from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer 

class TrainPipeline:
    def run (self,source_path:str):
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
            ) = transformation.initiate_data_transformation(train_path,test_path)

            # step-3 Model Trainer...
            trainer = ModelTrainer()
            f1_cat,f1_type = trainer.initiate_model_training(
                X_train, X_test,
                y_train_cat,  y_test_cat,
                y_train_type, y_test_type,
                class_weight_dict
            )
            logger.info(f"Final F1 — Category   : {f1_cat}")
            logger.info(f"Final F1 — Issue Type : {f1_type}")
            logger.info("========== TRAINING PIPELINE COMPLETED ==========")

            return f1_cat,f1_type
             

        except Exception as e:
            raise CustomException(e, sys)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str , required=True,help = "Path for Input Data")
    args = parser.parse_args()
    pipeline = TrainPipeline()
    f1_cat,f1_type = pipeline.run(args.source_path)
    print(f"\n Training complete!")
    print(f"   Category F1   : {f1_cat}")
    print(f"   Issue Type F1 : {f1_type}")