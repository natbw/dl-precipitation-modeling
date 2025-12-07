# src/deeplearning_models.py
# Deep learning models for precipitation forecasting

import torch
import torch.nn as nn
import numpy as np

def create_history_windows_torch(X, y, T=7, horizon=3):
    """
    Convert time series data into windowed input-output tensors for PyTorch models.
    
    Parameters
    ----------
    X : np.array
        Input features (e.g., daily precipitation or predictors)
        Shape: (num_days, num_features)
    y : np.array
        Target values (e.g., precipitation at future time steps)
        Shape: (num_days,)
    T : int, optional
        Number of past days to include in each input sequence (default is 7)
    horizon : int, optional
        Prediction horizon: how many days ahead to predict (default is 3)
    
    Returns
    -------
    X_window_tensor : torch.FloatTensor
        Tensor of shape (num_samples, T, num_features) containing input sequences
    y_window_tensor : torch.FloatTensor
        Tensor of shape (num_samples, 1) containing corresponding targets
    """
    X_window = []
    y_window = []
    num_days = len(X)

    for t in range(T-1, num_days-horizon):
        X_window.append(X[t-T+1 : t+1])
        y_window.append(y[t + horizon])

    X_window = np.array(X_window)
    y_window = np.array(y_window).reshape(-1, 1)

    X_window_tensor = torch.tensor(X_window, dtype=torch.float32)
    y_window_tensor = torch.tensor(y_window, dtype=torch.float32)

    return X_window_tensor, y_window_tensor


class LSTMModel(nn.Module):
    """
    LSTM-based model for sequence forecasting of precipitation.
    
    Parameters
    ----------
    input_dim : int
        Number of input features per time step
    hidden_dim : int, optional
        Number of hidden units in the LSTM (default is 64)
    num_layers : int, optional
        Number of stacked LSTM layers (default is 1)
    dropout : float, optional
        Dropout rate between LSTM layers (default is 0.0)
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        """
        Initialize the LSTM model architecture.
        """
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
        
        # Fully connected output layer (predicts 1 value per sequence)
        self.fc = nn.Linear(hidden_dim, 1)
        # Optional ReLU can be enabled if needed
        # self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the LSTM model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1) with predicted precipitation
        """
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        # x = self.relu(x)
        return x


class TransformerEncoderModel(nn.Module):
    """
    Transformer encoder model for precipitation forecasting.

    Parameters
    ----------
    input_dim : int
        Number of input features per time step.
    model_dim : int, optional
        Hidden model dimension (default 64).
    num_heads : int, optional
        Number of attention heads (default 4).
    num_layers : int, optional
        Number of encoder layers (default 2).
    output_dim : int, optional
        Output dimension (default 1).
    """

    def __init__(self, input_dim, model_dim=64, num_heads=4, num_layers=2, output_dim=1):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, model_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_embedding(x)
        x = x.permute(1, 0, 2)  # (seq, batch, model_dim)

        out = self.transformer_encoder(x)
        out = out[-1, :, :]      # last time step
        out = self.fc(out)
        return out

    