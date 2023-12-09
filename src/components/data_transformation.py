import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            numerical_columns = ['age', 'driving_license', 'region_code', 'previously_insured', 'annual_premium', 'policy_sales_channel', 'vintage']
            categorical_columns = ['gender', 'vehicle_age', 'vehicle_damage']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,temp_path,test_path):

        try:
            temp_df=pd.read_csv(temp_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read temp and test data completed")

            temp_df.columns = temp_df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
            test_df.columns = test_df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

            gender = {'Male': 0, 'Female': 1}
            vehicle_age = {'> 2 Years': 2, '1-2 Year': 1, '< 1 Year': 0}
            vehicle_damage = {'Yes': 1, 'No': 0}

            temp_df['gender'] = temp_df['gender'].map(gender)
            temp_df['vehicle_age'] = temp_df['vehicle_age'].map(vehicle_age)
            temp_df['vehicle_damage'] = temp_df['vehicle_damage'].map(vehicle_damage)

            test_df['gender'] = test_df['gender'].map(gender)
            test_df['vehicle_age'] = test_df['vehicle_age'].map(vehicle_age)
            test_df['vehicle_damage'] = test_df['vehicle_damage'].map(vehicle_damage)

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()
           
            target_column_name = "response"

            temp_df.drop('id', axis=1)
            test_df.drop('id', axis=1)

            #numerical_columns = ['age', 'driving_license', 'region_code', 'previously_insured', 'annual_premium', 'policy_sales_channel', 'vintage', 'response']

            logging.info("Train val split from temp dataset after oversampling")

            y = temp_df['response'].values
            X = temp_df.drop('response', axis=1)

            sme = SMOTEENN(random_state=42)
            X_res, y_res = sme.fit_resample(X, y)

            input_feature_train_df, input_feature_val_df, target_feature_train_df, target_feature_val_df = train_test_split(X_res, y_res, test_size=0.2, stratify=y_res, random_state=42)

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_val_arr = preprocessing_obj.transform(input_feature_val_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            val_arr = np.c_[
                input_feature_val_arr, np.array(target_feature_val_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                val_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)