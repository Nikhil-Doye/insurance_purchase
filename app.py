from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=int(request.form.get('Gender')),
            age=int(request.form.get('Age')),
            driving_license=float(request.form.get('Driving_License')),
            region_code=float(request.form.get('Region_Code')),
            previously_insured=int(request.form.get('Previously_Insured')),
            vehicle_age=int(request.form.get('vehicle_age')),
            vehicle_damage=float(request.form.get('vehicle_damage')),
            annual_premium=float(request.form.get('Annual_Premium')),
            policy_sales_channel=float(request.form.get('Policy_Sales_Channel')),
            vintage=int(request.form.get('Vintage'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        
        print("after Prediction")
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0") 