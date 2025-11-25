# src/train.py
# Train models

from src.baseline_models import *
from src.evaluate import evaluate_predictions, evaluate_extreme_events
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

def load_data(data_path):
    npz = np.load(data_path, allow_pickle=True)
    X_train = npz["X_train"]
    y_train = npz["y_train"]
    dates_train = np.array(pd.to_datetime(npz["dates_train"]))

    X_test = npz["X_test"]
    y_test = npz["y_test"]
    dates_test = np.array(pd.to_datetime(npz["dates_test"]))

    return X_train, y_train, dates_train, X_test, y_test, dates_test

def train_baseline_models(data_path, T=7, horizon=3, p_window=7):

    # LOAD DATA
    X_train, y_train, dates_train, X_test, y_test, dates_test = load_data(data_path)

    X_train_window, y_train_window = create_history_windows(X_train, y_train, T=T, horizon=horizon)
    X_test_window, y_test_window = create_history_windows(X_test, y_test, T=T, horizon=horizon)

    results = {}

    # TRAIN CLIMATOLOGY MODEL
    climatology = ClimatologyModel()
    climatology.fit(y_train, dates_train)
    y_pred_cl = climatology.predict(dates_test)
    results['climatology'] = {
        'y_pred': y_pred_cl,
        'rmse': root_mean_squared_error(y_test, y_pred_cl),
        'mae': mean_absolute_error(y_test, y_pred_cl)
    }

    # TRAIN PERSISTENCE MODEL
    persistence = PersistenceModel(window=p_window)
    persistence.fit(y_train)
    y_pred_ps = persistence.predict(y_test_window)
    results['persistence'] = {
        'y_pred': y_pred_ps,
        'rmse': root_mean_squared_error(y_test_window, y_pred_ps),
        'mae': mean_absolute_error(y_test_window, y_pred_ps)
    }

    # TRAIN LINEAR REGRESSION MODEL
    linreg = LinearRegressionModel()
    linreg.fit(X_train_window, y_train_window)
    y_pred_lr = linreg.predict(X_test_window)
    results['linear_regression'] = {
        'y_pred': y_pred_lr,
        'rmse': root_mean_squared_error(y_test_window, y_pred_lr),
        'mae': mean_absolute_error(y_test_window, y_pred_lr)
    }

    return results

def print_results(results):
    print("Baseline Results:")
    for model, result in results.items():
        print(f"{model}: RMSE = {result['rmse']}, MAE = {result['mae']}")
