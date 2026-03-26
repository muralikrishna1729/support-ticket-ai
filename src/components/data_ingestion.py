import pandas as pd 
import os 
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.exception import CustomException 
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","data.csv")
    train_data_path : str = os.path.join("artifacts", "train.csv")
    test_data_path  : str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
    def  initiate_data_ingestion(self,source_path):
        logger.info("Data Ingestion was Started...")
        try:
            df = pd.read_csv(source_path)
            logger.info(f"Raw data loaded: {df.shape}")

            #-- now filter and take only EN queries
            df = df[df["language"]== "en"].reset_index(drop = True)
            logger.info(f"After Engilsh Filter data loaded: {df.shape}")

            df['text'] = (df['subject'].fillna(" ")+" "+df["body"].fillna(" ")).str.strip()

            df = df[["text",'queue','type']].rename(columns = {"queue":"category", "type":"issue_type"})

            logger.info(f"Dropping the Empty : {df.shape}")
            df = df[df["text"].str.strip() != ""].reset_index(drop = True)
            logger.info(f"Final shape after cleaning: {df.shape}")

            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            df.to_csv(self.config.raw_data_path, index= False)


            train_df, test_df = train_test_split(
                df,
                test_size    = 0.15,
                random_state = 42,
                stratify     = df['category']
            )
            train_df.to_csv(self.config.train_data_path,index = False)
            test_df.to_csv(self.config.test_data_path,index = False)
            logger.info(f"Train size : {len(train_df)}")
            logger.info(f"Test size  : {len(test_df)}")
            logger.info("=== Data Ingestion Completed ===")

            return (
                self.config.train_data_path,
                self.config.test_data_path
            )


        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion("notebook/data/tickets-dataset.csv")
    logger.info("Data Ingestion Completed.")

    data_transformation = DataTransformation()
    X_train, X_test, y_train_cat,  y_test_cat,y_train_type, y_test_type,class_weight_dict,le_category, le_issue_type  = data_transformation.initiate_data_transformation(train_data,test_data)
    logger.info("Data Transformation Completed.")