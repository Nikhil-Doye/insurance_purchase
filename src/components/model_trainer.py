import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from dataclasses import dataclass
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import f1_score
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','Model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,val_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and val data')
            X_train, y_train, X_val, y_val = (
                train_array[:,:-1],
                train_array[:,-1],
                val_array[:,:-1],
                val_array[:,-1])
            
            models = {
                'Decision Tree':DecisionTreeClassifier(),
                'Random Forest Classfier':RandomForestClassifier(),
                'AdaBoost':AdaBoostClassifier(),
                'XGBoost':XGBClassifier(),
                'LightGBM':LGBMClassifier(),
                'CatBoost':CatBoostClassifier()
                }
            
            model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, models=models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {model_report}')

            # To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            print(f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Accuracy Score: {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
            
            predicted = best_model.predict(X_val)

            f1_score = f1_score(y_val, predicted)
            return f1_score

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)    