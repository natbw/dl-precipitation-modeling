# src/deeplearning_models.py
# Deep learning models for precipitation forecasting

import torch
import torch.nn as nn
import numpy as np

def create_history_windows_torch(X, y, T=7, horizon=3):

    X_window = []
    y_window = []
    num_days = len(X)

    for t in range(T-1, num_days-horizon):
        X_window.append(X[t-T+1 : t+1])
        y_window.append(y[t + horizon])

    X_window = np.array(X_window)
    y_window = np.array(y_window)
    X_window_tensor = torch.tensor(X_window, dtype=torch.float32)
    y_window_tensor = torch.tensor(y_window, dtype=torch.float32)
    
    return X_window_tensor, y_window_tensor


class LSTMModel(nn.Model):
    def __init__(self):
        super
    

