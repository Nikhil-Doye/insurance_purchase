import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from xgboost import XGBClassifier
from dataclasses import dataclass
from catboost import CatBoostClassifier
#from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import f1_score,roc_curve, auc, precision_recall_curve
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models,plot_curves, hp_tuning, plot_feature_importance

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','Model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initate_model_training(self,train_array,val_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and val data')
            X_train, y_train, X_val, y_val, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                val_array[:,:-1],
                val_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1])
            
            train_models = {
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'XGBoost': XGBClassifier(),
                #'LightGBM':LGBMClassifier(),
                'CatBoost': CatBoostClassifier(verbose=False)
                }
            
            train_model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val, models=train_models)
            print(train_model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report: {train_model_report}')

            # To get the best model score from the dictionary
            best_train_model_score = max(sorted(train_model_report.values()))

            best_train_model_name = list(train_model_report.keys())[
                list(train_model_report.values()).index(best_train_model_score)
            ]
            
            best_train_model = train_models[best_train_model_name]

            print(f'Best Model Found, Model Name: {best_train_model}, F1 Score: {best_train_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_train_model}, F1 Score: {best_train_model_score}')
            
            plot_curves(models_dict=train_models, X=X_train, y=y_train)
            print('\n====================================================================================\n')
            logging.info(f'Plotting training curves')

            val_models = {
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'XGBoost': XGBClassifier(),
                #'LightGBM':LGBMClassifier(),
                'CatBoost': CatBoostClassifier(verbose=False)
                }
            
            param={
                "Decision Tree": {
                    'criterion': ['gini', 'entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': [None, 'balanced']
                },
                "Random Forest":{
                    'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
                    'max_features': ['sqrt', 'log2'],  # Maximum number of features to consider for splitting a node
                    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
                    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
                    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
                    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
                },
                "AdaBoost":{
                    'n_estimators': [50, 100, 200],  # Number of weak learners (trees) to train
                    'learning_rate': [0.01, 0.1, 0.2],  # Weight applied to each weak learner
                    'algorithm': ['SAMME', 'SAMME.R']  # AdaBoost algorithm to use
                },
                "XGBoost":{
                    'learning_rate': [0.01, 0.1, 0.2, 0.3],
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2, 0.3],
                    'reg_alpha': [0, 0.1, 0.5, 1.0],
                    'reg_lambda': [0, 0.1, 0.5, 1.0],
                    'scale_pos_weight': [1, 2, 3]
                },
                "CatBoost":{
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'depth': [4, 6, 8, 10],
                    'l2_leaf_reg': [1, 3, 5, 7, 9],
                    'iterations': [500, 1000, 1500],
                    'border_count': [32, 64, 128],
                    'thread_count': [2, 4, 8],
                    'loss_function': ['Logloss', 'CrossEntropy'],
                    'eval_metric': ['Logloss', 'AUC']
                }                
            }

            val_model_report = hp_tuning(X_train, y_train, X_val, y_val, X_test, y_test, models=val_models, param=param)
            print(val_model_report)

            # To get the best model score from the dictionary
            best_val_model_score = max(sorted(val_model_report.values()))

            best_val_model_name = list(val_model_report.keys())[
                list(val_model_report.values()).index(best_val_model_score)
            ]
            
            best_val_model = val_models[best_val_model_name]

            print(f'Best Model Found, Model Name: {best_val_model}, F1 Score: {best_val_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_val_model}, F1 Score: {best_val_model_score}')

            plot_curves(models_dict=val_models, X=X_val, y=y_val)
            print('\n====================================================================================\n')
            logging.info(f'Plotting Validation curves')

            print(plot_feature_importance(models_dict=train_models,X_val=X_train))
            logging.info(f'Plotting feature_importance curves')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_train_model
            )
            
            
            predicted = best_train_model.predict(X_test)

            # # Compute ROC curve and ROC area
            # fpr, tpr, _ = roc_curve(y_test, predicted)
            # roc_auc = auc(fpr, tpr)

            # # Plot ROC curve
            # plt.figure(figsize=(8, 6))
            # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('BEST ROC CURVE')
            # plt.legend(loc="lower right")
            # plt.show()

            # # Compute PR curve and area under the curve
            # precision, recall, _ = precision_recall_curve(y_test, predicted)
            # pr_auc = auc(recall, precision)

            # # Plot PR curve
            # plt.figure(figsize=(8, 6))
            # plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title('BEST PR CURVE')
            # plt.legend(loc="lower right")
            # plt.show()

            f1 = f1_score(y_test, predicted)
            return f1

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)