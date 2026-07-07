import pandas as pd 
import os
import sys
from src.exception import CustomException
from src.logger import logger
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts",'train.csv')
    test_data_path:str = os.path.join("artifacts",'test.csv')
    raw_data_path:str = os.path.join('artifacts','data.csv')
    source_data_path:str = os.path.join('notebook/data','tickets-dataset.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, source_path: str):
        logger.info("=== Data Ingestion Started ===")
        logger.info("Entered the Data ingestion method or component")
        try:
            df = pd.read_csv(source_path)
            if df.empty:
                raise Exception("Dataset is empty")
            logger.info("Read the dataset as dataframe")
            logger.info(f"Dataset shape: {df.shape}")
            df = df[df["language"]=="en"].reset_index(drop=True)
            logger.info(f"After English filter: {df.shape}")

            # Combine subject + body
            df["text"] = (df["subject"].fillna('')+' '+df['body'].fillna('')).str.strip()

            # Select relevant columns
            df = df[['text', 'queue', 'type']].rename(columns={
                'queue' : 'category',
                'type'  : 'issue_type'
            })

            df = df[df['text'].str.strip() != ''].reset_index(drop=True)
            logger.info(f"Final shape: {df.shape}")
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index = False, header= True)


            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            # Creating Train and Test csv files and save it in data paths.
            logger.info("Train test split Initiated")

            train_set,test_set = train_test_split(df,test_size = 0.15, random_state = 42,stratify = df['category'])
            train_set.to_csv(self.ingestion_config.train_data_path, index = False , header  =True)
            test_set.to_csv(self.ingestion_config.test_data_path,index = False , header = True)
            logger.info(f"Train: {len(train_set)} | Test: {len(test_set)}")
            logger.info("=== Data Ingestion Completed ===")
            logger.info("Ingestion of train and test data completed")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    source_path = obj.ingestion_config.source_data_path
    obj.initiate_data_ingestion(source_path)