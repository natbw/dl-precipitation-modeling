# src/evaluate.py
# Evaluate models

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_predictions(y_true, y_pred, model="Model"):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"{model} -> RMSE: {rmse}, MAE: {mae}")
    return rmse, mae

def evaluate_extreme_events(y_true, y_pred, percentile=95):
    threshold = np.percentile(y_true, percentile)
    mask = y_true >= threshold
    rmse_extreme = mean_squared_error(y_true[mask], y_pred[mask], squared=False)
    mae_extreme = mean_absolute_error(y_true[mask], y_pred[mask])
    count = np.sum((y_pred[mask] >= threshold))
    total_extreme = np.sum(mask)
    print(f"Extreme events >{percentile}th percentile: RMSE: {rmse_extreme}, MAE: {mae_extreme}")
    print(f"Total extreme events: {total_extreme}, Total extreme events predicted: {count}")
    return rmse_extreme, mae_extreme, count, total_extreme