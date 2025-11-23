# src/train.py
# Train models

from src.baseline_models import ClimatologyModel, PersistenceModel, LinearRegressionModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import pickle

def train_baseline_models(
        X_train,
        y_train,
        dates_train,
        X_val,
        y_val,
        dates_val,
        X_test,
        y_test,
        dates_test
        ):
    
    results = {}
    
    climatology = ClimatologyModel()
    climatology.fit(y_train, dates_train)
    y_preds_cl = climatology.predict(dates_test)
    results['climatology'] = {
        'y_pred' : y_preds_cl,
        'rmse' : mean_squared_error(y_test, y_preds_cl, squared=False),
        'mae' : mean_absolute_error(y_test, y_preds_cl)
        }
    
    persistence = PersistenceModel()
    y_pred_ps = persistence.predict(y_train[-len(y_test):])
    results['persistence'] = {
        'y_pred' : y_pred_ps,
        'rmse' : mean_squared_error(y_test, y_pred_ps, squared=False),
        'mae' : mean_absolute_error(y_test, y_pred_ps)
        }
    
    linreg = LinearRegressionModel()
    linreg.fit(X_train, y_train)
    y_pred_lr = linreg.predict(X_test)
    results['linear_regression'] = {
        'y_pred': y_pred_lr,
        'rmse': mean_squared_error(y_test, y_pred_lr, squared=False),
        'mae': mean_absolute_error(y_test, y_pred_lr)
        }
    
def print_results(results):
    print("Baseline Results:")
    for model, result in results.items():
        print(f"{model}: RMSE = {result['rmse']}, MAE = {result['mae']}")

