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

    return X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test, feature_names

def train_baseline_models(data_path, T=7, horizon=3, p_window=7):

    # LOAD DATA
    X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test, feature_names = load_data(data_path)

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

def quantile_loss(y_pred, y_true, q=0.5):
    e = y_true - y_pred
    return torch.mean(torch.where(e >= 0, q * e, (q - 1) * e))

def log_mse_loss(y_pred, y_true):
    y_pred = torch.clamp(y_pred, min=0.0)
    y_true = torch.clamp(y_true, min=0.0)
    return torch.nn.MSELoss()(torch.log1p(y_pred), torch.log1p(y_true))

def train_lstm_model(
    data_path,
    T=7,
    horizon=3,
    hidden_dim=64,
    num_layers=1,
    dropout=0.0,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    device='cuda',
    patience=10,
    weight_decay=1e-4,
    loss_type='rmse',
    quantile_q=0.5
):
    
    # LOAD DATA
    X_train, y_train, dates_train, X_val, y_val, dates_val, X_test, y_test, dates_test, feature_names = load_data(data_path)

    X_train_window, y_train_window = create_history_windows_torch(X_train, y_train, T=T, horizon=horizon)
    X_val_window, y_val_window = create_history_windows_torch(X_val, y_val, T=T, horizon=horizon)
    X_test_window, y_test_window = create_history_windows_torch(X_test, y_test, T=T, horizon=horizon)

    input_dim = X_train_window.shape[2]

    train_dataset = data.TensorDataset(X_train_window, y_train_window)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = data.TensorDataset(X_val_window, y_val_window)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    if loss_type == 'rmse' or loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_type == 'mae':
        criterion = torch.nn.L1Loss()
    elif loss_type == 'huber':
        criterion = torch.nn.SmoothL1Loss(beta=1.0)
    elif loss_type == 'quantile':
        criterion = lambda pred, true: quantile_loss(pred, true, q=quantile_q)
    elif loss_type == 'log_mse':
        criterion = log_mse_loss
    else:
        raise ValueError(f"Loss type: {loss_type} not a valid option.")

    train_losses = []
    val_losses = []
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0

    for ep in range(epochs):
        # TRAINING
        model.train()
        total_train_sq_error = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            if loss_type == 'rmse' or loss_type == 'mse':
                total_train_sq_error += torch.nn.functional.mse_loss(y_pred, y_batch, reduction='sum').item()
            else:
                total_train_sq_error += loss.item() * X_batch.size(0)

        avg_train_loss = total_train_sq_error / len(train_dataset)
        if loss_type == 'rmse':
            avg_train_loss = np.sqrt(avg_train_loss)
        train_losses.append(avg_train_loss)

        # VALIDATION
        model.eval()
        total_val_sq_error = 0.0
        total_val_loss = 0.0
        count = 0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_pred = model(X_val_batch)
                val_loss = criterion(val_pred, y_val_batch)
                total_val_loss += val_loss.item() * X_val_batch.size(0)
                total_val_sq_error += torch.nn.functional.mse_loss(val_pred, y_val_batch, reduction='sum').item()
                count += X_val_batch.size(0)

        avg_val_loss = total_val_loss / count
        if loss_type == 'rmse':
            avg_val_loss = np.sqrt(avg_val_loss)
        val_losses.append(avg_val_loss)

        val_rmse = np.sqrt(total_val_sq_error / count)
        scheduler.step(val_rmse)

        # EARLY STOPPING
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {ep + 1}")
            break

        if ep == 0 or (ep + 1) % 25 == 0:
            print(f"Epoch {ep+1}/{epochs} | Train Loss ({loss_type.upper()}): {avg_train_loss} | Val Loss ({loss_type.upper()}): {avg_val_loss} | Val RMSE: {val_rmse}")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_window.to(device)).cpu().numpy()

    lstm_rmse = root_mean_squared_error(y_test_window.numpy(), y_pred_test)
    lstm_mae = mean_absolute_error(y_test_window.numpy(), y_pred_test)

    results = {
        "LSTM": {
            "y_pred": y_pred_test.flatten(),
            "rmse": lstm_rmse,
            "mae": lstm_mae,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "loss_type": loss_type.upper(),
            "model": model
        }
    }

    return results

def print_results(results):
    print("Test Data Results:")
    print("------------------")
    
    for model_name, result in results.items():
        name = model_name.replace("_", " ").upper()
        rmse = result.get("rmse", "N/A")
        mae = result.get("mae", "N/A")
        print(f"{name} --> RMSE = {rmse}, MAE = {mae}")
