# src/baseline_models.py
# Baseline linear regression models for precipitation forecasting

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

# Simple Baseline Models
# climatology --> predict (yt+h) as the mean daily precipitation for that calendar day over the training years
# persistence --> predict (yt+h = yt) or average over last few days
# linear regression --> flattened feature vector zt = vec(xt-T+1)


def create_history_windows(X, y, T=7, horizon=3):

    X_window = []
    y_window = []
    num_days = len(X)

    for t in range(T-1, num_days-horizon):
        X_window.append(X[t-T+1 : t+1])
        y_window.append(y[t + horizon])

    return np.array(X_window), np.array(y_window)


class ClimatologyModel:
    
    def __init__(self):
        self.day_mean = None
        
    def fit(self, y_train, dates_train):
        day_of_year = np.array([pd.to_datetime(d).timetuple().tm_yday for d in dates_train])
        unique_days = np.unique(day_of_year)
        self.day_mean = {day: np.mean(y_train[day_of_year == day]) for day in unique_days}
        
    def predict(self, dates_test):
        day_of_year = np.array([pd.to_datetime(d).timetuple().tm_yday for d in dates_test])
        return np.array([self.day_mean[day] for day in day_of_year])


class PersistenceModel:
    
    def __init__(self, window=7):
        self.window = window
        self.y_train = None
    
    def fit(self, y_train):
        self.y_train = y_train
    
    def predict(self, y_test_window):
        y_pred = []
        history = list(self.y_train[-self.window:])

        for i in range(len(y_test_window)):
            y_next = np.mean(history[-self.window:])
            y_pred.append(y_next)
            history.append(y_test_window[i])
        
        return np.array(y_pred)
    

class LinearRegressionModel:
    
    def __init__(self):
        self.model = LinearRegression()
        self.T = None
        self.d = None
        
    def fit(self, X_train, y_train):
        self.T = X_train.shape[1]
        self.d = X_train.shape[2]

        X_flat = X_train.reshape(len(X_train), self.T * self.d)
        self.model.fit(X_flat, y_train)
        
    def predict(self, X_test):
        X_flat = X_test.reshape(len(X_test), self.T * self.d)
        return self.model.predict(X_flat)

