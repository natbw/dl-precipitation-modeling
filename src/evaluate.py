# src/evaluate.py
# Evaluate models

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_predictions(y_true, y_pred, model="Model"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"{model} -> RMSE: {rmse}, MAE: {mae}")  # rmse:.4f
    
    return rmse, mae

def evaluate_extreme_events(y_true, y_pred, percentile=95):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    threshold = np.percentile(y_true, percentile)
    mask = y_true >= threshold
    
    if np.sum(mask) == 0:
        print(f"No extreme events above {percentile}th percentile.")
        return None, None, 0, 0
    
    rmse_extreme = root_mean_squared_error(y_true[mask], y_pred[mask])
    mae_extreme = mean_absolute_error(y_true[mask], y_pred[mask])
    count = np.sum(y_pred[mask] >= threshold)
    total_extreme = np.sum(mask)
    
    print(f"Extreme events > {percentile}th percentile -> RMSE: {rmse_extreme}, MAE: {mae_extreme}")
    print(f"Total extreme events: {total_extreme}, Total extreme events predicted: {count}")

    return rmse_extreme, mae_extreme, count, total_extreme
    