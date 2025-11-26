# src/train.py
# Train models

from src.baseline_models import *
from src.evaluate import evaluate_predictions, evaluate_extreme_events
from src.deeplearning_models import LSTMModel, create_history_windows_torch
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import torch
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import numpy as np

def load_data(data_path):
    npz = np.load(data_path, allow_pickle=True)

    X_train = npz["X_train"]
    y_train = npz["y_train"]
    dates_train = np.array(pd.to_datetime(npz["dates_train"]))

    X_val = npz["X_val"]
    y_val = npz["y_val"]
    dates_val = np.array(pd.to_datetime(npz["dates_val"]))

    X_test = npz["X_test"]
    y_test = npz["y_test"]
    dates_test = np.array(pd.to_datetime(npz["dates_test"]))

    feature_names = npz["feature_names"]

    return (X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test, feature_names)

def train_baseline_models(data_path, T=7, horizon=3, p_window=7):

    # LOAD DATA
    (X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test, feature_names) = load_data(data_path)

    X_train_window, y_train_window = create_history_windows(X_train, y_train, T=T, horizon=horizon)
    X_test_window, y_test_window = create_history_windows(X_test, y_test, T=T, horizon=horizon)

    results = {}

    # TRAIN CLIMATOLOGY MODEL
    climatology = ClimatologyModel()
    climatology.fit(y_train, dates_train)
    y_pred_cl = climatology.predict(dates_test)
    results["climatology"] = {
        "y_pred": y_pred_cl,
        "rmse": root_mean_squared_error(y_test, y_pred_cl),
        "mae": mean_absolute_error(y_test, y_pred_cl)
    }

    # TRAIN PERSISTENCE MODEL
    persistence = PersistenceModel(window=p_window)
    persistence.fit(y_train)
    y_pred_ps = persistence.predict(y_test_window, horizon=horizon)
    results["persistence"] = {
        "y_pred": y_pred_ps,
        "rmse": root_mean_squared_error(y_test_window, y_pred_ps),
        "mae": mean_absolute_error(y_test_window, y_pred_ps)
    }

    # TRAIN LINEAR REGRESSION MODEL
    linreg = LinearRegressionModel()
    linreg.fit(X_train_window, y_train_window)
    y_pred_lr = linreg.predict(X_test_window)
    results["linear_regression"] = {
        "y_pred": y_pred_lr,
        "rmse": root_mean_squared_error(y_test_window, y_pred_lr),
        "mae": mean_absolute_error(y_test_window, y_pred_lr)
    }

    return results

def train_lstm_model(
    data_path,
    T=7,
    horizon=3,
    hidden_dim=64,
    num_layers=1,
    dropout=0.0,
    epochs=20,
    batch_size=64,
    lr=1e-3,
    device='cuda'
):
    
    # LOAD DATA
    (X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test, feature_names) = load_data(data_path)

    X_train_window, y_train_window = create_history_windows_torch(X_train, y_train, T=T, horizon=horizon)
    X_val_window, y_val_window = create_history_windows_torch(X_val, y_val, T=T, horizon=horizon)
    X_test_window, y_test_window = create_history_windows_torch(X_test, y_test, T=T, horizon=horizon)

    input_dim = X_train_window.shape[2]

    train_dataset = data.TensorDataset(X_train_window, y_train_window)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    results = {}

    for ep in range(epochs):
        model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_train_rmse = np.sqrt(total_loss / len(X_train_window))
        train_losses.append(avg_train_rmse)
            
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_window.to(device))
            val_mse = criterion(val_pred.cpu(), y_val_window).item()
            val_rmse = np.sqrt(val_mse)
        val_losses.append(val_rmse)

        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs} | Train RMSE: {avg_train_rmse} | Val RMSE: {val_rmse}")

    with torch.no_grad():
        y_pred_test = model(X_test_window.to(device)).cpu().numpy()

    lstm_rmse = root_mean_squared_error(y_test_window.numpy(), y_pred_test)
    lstm_mae  = mean_absolute_error(y_test_window.numpy(), y_pred_test)

    results = {
        "LSTM": {
            "y_pred": y_pred_test.flatten(),
            "rmse": lstm_rmse,
            "mae": lstm_mae,
            "train_rmse": train_losses,
            "val_rmse": val_losses,
            "model": model
        }
    }

    return results

def print_results(results):
    print("Training Results:")
    print("------------------")

    for model_name, result in results.items():
        name = model_name.replace("_", " ").upper()
        rmse = result.get("rmse", "N/A")
        mae  = result.get("mae", "N/A")

        print(f"{name} --> RMSE = {rmse}, MAE = {mae}")
