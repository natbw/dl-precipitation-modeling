# src/evaluate.py
# Evaluate models

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_predictions(y_true, y_pred, model="Model"):
    """
    Evaluate standard regression metrics (RMSE and MAE) for a model.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values from the model.
    model : str, optional
        Name of the model (default "Model") for printing results.
    
    Returns
    -------
    rmse : float
        Root Mean Squared Error.
    mae : float
        Mean Absolute Error.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = root_mean_squared_error(y_true, y_pred) # root mean square error
    mae  = mean_absolute_error(y_true, y_pred) # mean absolute error
    print(f"{model} -> RMSE: {rmse}, MAE: {mae}")  # rmse:.4f
    
    return rmse, mae

def evaluate_extreme_events(y_true, y_pred, percentile=95):
    """
    Evaluate model performance specifically for extreme events above a given percentile.
    
    Metrics include RMSE and MAE for extreme events, as well as predicted vs total extreme events.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values from the model.
    percentile : float, optional
        Percentile threshold to define extreme events (default 95).
    
    Returns
    -------
    rmse_extreme : float or None
        RMSE for extreme events. None if no extreme events are found.
    mae_extreme : float or None
        MAE for extreme events. None if no extreme events are found.
    predicted_extreme_count : int
        Number of predicted extreme events above threshold.
    total_extreme : int
        Total number of true extreme events above threshold.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    threshold = np.percentile(y_true, percentile)
    mask = y_true >= threshold
    
    total_extreme = np.sum(mask)
    if total_extreme == 0:
        print(f"No extreme events above {percentile}th percentile.")
        return None, None, 0, 0
    
    rmse_extreme = root_mean_squared_error(y_true[mask], y_pred[mask])
    mae_extreme = mean_absolute_error(y_true[mask], y_pred[mask])
    predicted_extreme_count = np.sum(y_pred[mask] >= threshold)
    
    print(f"Extreme events > {percentile}th percentile -> RMSE: {rmse_extreme}, MAE: {mae_extreme}")
    print(f"Total extreme events: {total_extreme}, Predicted extreme events: {predicted_extreme_count}")

    return rmse_extreme, mae_extreme, predicted_extreme_count, total_extreme
    