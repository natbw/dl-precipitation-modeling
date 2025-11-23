# src/baseline_models.py
# Baseline linear regression models for precipitation forecasting

import numpy as np
from sklearn.linear_model import LinearRegression

# Simple Baseline Models
# climatology --> predict (yt+h) as the mean daily precipitation for that calendar day over the training years
# persistence --> predict (yt+h = yt) or average over last few days
# linear regression --> flattened feature vector zt = vec(xt-T+1)

class ClimatologyModel:
    
    def __init__(self):
        self.day_mean = None
        
    def fit(self, y_train, dates_train):
        day_of_year = np.array([d.timetuple().tm_yday for d in dates_train])
        self.day_mean = {day: np.mean(y_train[day_of_year == day]) for day in np.unique(day_of_year)}
        
    def predict(self, dates_test):
        day_of_year = np.array([d.timetuple().tm_yday for d in dates_test])
        return np.array([self.day_mean[day] for day in day_of_year])


class PersistenceModel:
    
    def __init__(self):
        pass
    
    def fit(self, y_train, X_train):
        pass
    
    def predict(self, y_previous):
        return y_previous
    

class LinearRegressionModel:
    
    def __init(self):
        self.model = LinearRegression()
        
    def fit(self, X_train, y_train):
        n_samples, T, n_features = X_train.shape
        X_flat = X_train.reshape(n_samples, T * n_features)
        self.model.fit(X_flat, y_train)
        
    def predict(self, X_test):
        n_samples, T, n_features = X_test.shape
        X_flat = X_test.reshape(n_samples, T * n_features)
        return self.model.predict(X_flat)

