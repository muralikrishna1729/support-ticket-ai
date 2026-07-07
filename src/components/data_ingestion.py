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
    source_data_path: str = os.path.join('notebook/data', 'dataset-tickets-multi-lang-4-20k.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self, source_path: str):
        logger.info("=== Data Ingestion Started ===")
        try:
            # Load primary dataset
            df = pd.read_csv(source_path)
            logger.info(f"Primary dataset: {df.shape}")

            # Combine with 4k multilang dataset
            extra_path = "notebook/data/dataset-tickets-multi-lang3-4k.csv"
            if os.path.exists(extra_path):
                df_extra = pd.read_csv(extra_path)
                required = ['language', 'queue', 'type', 'body', 'subject']
                if all(col in df_extra.columns for col in required):
                    df = pd.concat([df, df_extra], ignore_index=True)
                    logger.info(f"Combined with 4k dataset → {df.shape}")

            # Drop duplicates
            before = len(df)
            df = df.drop_duplicates(subset=['body']).reset_index(drop=True)
            logger.info(f"Dropped {before - len(df)} duplicates → {df.shape}")

            # Filter English only
            df = df[df['language'] == 'en'].reset_index(drop=True)
            logger.info(f"After English filter: {df.shape}")

            # Combine subject + body
            df['text'] = (
                df['subject'].fillna('') + ' ' +
                df['body'].fillna('')
            ).str.strip()

            # Select relevant columns
            df = df[['text', 'queue', 'type']].rename(columns={
                'queue' : 'category',
                'type'  : 'issue_type'
            })

            # Drop empty rows
            df = df[df['text'].str.strip() != ''].reset_index(drop=True)
            logger.info(f"Final shape: {df.shape}")

            # Save raw
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split
            train_set, test_set = train_test_split(
                df,
                test_size    = 0.15,
                random_state = 42,
                stratify     = df['category']
            )

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,   index=False, header=True)

            logger.info(f"Train: {len(train_set)} | Test: {len(test_set)}")
            logger.info("=== Data Ingestion Completed ===")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
if __name__ == "__main__":
    obj = DataIngestion()
    source_path = obj.ingestion_config.source_data_path
    obj.initiate_data_ingestion(source_path)