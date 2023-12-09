import os
import sys

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from itertools import cycle
import dill
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def eval_metrics(actual,pred):
        accuracy = accuracy_score(actual,pred)
        precision = precision_score(actual,pred)
        recall = recall_score(actual,pred)
        f1 = f1_score(actual,pred)
        return accuracy, precision, recall, f1

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train, y_train)  # Train model

            y_test_pred = model.predict(X_test)

            (test_accuracy, test_precision, test_recall, test_f1) = eval_metrics(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_f1

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def plot_curves(models_dict, X, y):
    """
    Plot ROC and PR curves for different models.

    Parameters:
    - models_dict (dict): A dictionary containing model names as keys and model objects as values.
    - X_test (array-like): Feature matrix for the test set.
    - y_test (array-like): True labels for the test set.
    """
    try:
        # Set up plot
        plt.figure(figsize=(12, 6))
        num_models = len(models_dict)
        fig, axes = plt.subplots(nrows=num_models, ncols=2, figsize=(12, 6 * num_models))
    
        for i, (model_name, model) in enumerate(models_dict.items()):
            ax_roc = axes[i, 0]
            ax_pr = axes[i, 1]

            # ROC Curve
            y_prob = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)

            ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')

            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title(f'ROC Curve - {model_name}')
            ax_roc.legend()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y, y_prob)
            avg_precision = average_precision_score(y, y_prob)

            ax_pr.plot(recall, precision, label=f'{model_name} (Avg Precision = {avg_precision:.2f})')
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.set_title(f'Precision-Recall Curve - {model_name}')
            ax_pr.legend()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        raise CustomException(e, sys)

def hp_tuning(X_train, y_train, X_val, y_val, X_test, y_test, models, param):
    try:
        final_report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = RandomizedSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_val, y_val)

            y_test_pred = model.predict(X_test)

            (test_accuracy, test_precision, test_recall, test_f1) = eval_metrics(y_test,y_test_pred)

            final_report[list(models.keys())[i]] =  test_f1

        return final_report

    except Exception as e:
        raise CustomException(e, sys)

def plot_feature_importance(models_dict, X_val):
    num_models = len(models_dict)
    plt.figure(figsize=(15, 5 * num_models))

    for i, (model_name, model) in enumerate(models_dict.items(), start=1):
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.subplot(num_models, 2, 2 * i - 1)
            plt.bar(range(X_val.shape[1]), importances[indices],
                    align="center", color="skyblue")
            plt.xticks(range(X_val.shape[1]), indices)
            plt.xlabel("Feature Index")
            plt.ylabel("Feature Importance")
            plt.title(f"{model_name} - Feature Importances")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            raise CustomException(e, sys)