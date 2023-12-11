import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    temp_data_path: str=os.path.join('artifacts',"temp.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"dataset.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\dataset.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.temp_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            y_set = df['Response'].values
            #x_set = df.drop('Response', axis=1)

            logging.info("Train test split initiated")
            temp_set,test_set=train_test_split(df,test_size=0.2,random_state=42, stratify=y_set)

            temp_set.to_csv(self.ingestion_config.temp_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.temp_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    temp_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,val_arr,test_arr,_=data_transformation.initiate_data_transformation(temp_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initate_model_training(train_arr,val_arr,test_arr))


