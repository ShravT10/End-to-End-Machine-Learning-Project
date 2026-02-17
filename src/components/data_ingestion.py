import os
import sys

from src.logger import logging
from src.exception import CustomException

import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTranformationConfig
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer 

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion method/component")
        try:
            df = pd.read_csv("notebook\data\data.csv")
            # logging.info("Loaded Data as dataframe")

            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)
            
            # logging.info("Train Test Split initiated")
            train_set , test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)
            
            # logging.info("Ingestion Complete")

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__ == '__main__':
    obj = DataIngestion()
    train_data , test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()

    train , test , _ = data_transformation.initiate_data_transfromation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train,test))



