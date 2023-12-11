import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","Model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')

            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender,
        age,
        driving_license,
        region_code,
        previously_insured,
        vehicle_age,
        vehicle_damage,
        annual_premium,
        policy_sales_channel,
        vintage):

        self.gender = gender

        self.age = age

        self.driving_license = driving_license

        self.region_code = region_code

        self.previously_insured = previously_insured

        self.vehicle_age = vehicle_age

        self.vehicle_damage = vehicle_damage

        self.annual_premium = annual_premium

        self.policy_sales_channel = policy_sales_channel

        self.vintage = vintage

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "age": [self.age],
                "driving_license": [self.driving_license],
                "region_code": [self.region_code],
                "previously_insured": [self.previously_insured],
                "vehicle_age": [self.vehicle_age],
                "vehicle_damage": [self.vehicle_damage],
                "annual_premium": [self.annual_premium],
                "policy_sales_channel": [self.policy_sales_channel],
                "vintage": [self.vintage]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
