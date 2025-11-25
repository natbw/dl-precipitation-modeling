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
    y_window = np.array(y_window).reshape(-1,1)
    X_window_tensor = torch.tensor(X_window, dtype=torch.float32)
    y_window_tensor = torch.tensor(y_window, dtype=torch.float32)
    
    return X_window_tensor, y_window_tensor


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# class TransformerModel(nn.Module):
#     pass
    