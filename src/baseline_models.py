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
    """
    Convert time series data into supervised learning input-output windows.

    Parameters
    ----------
    X : np.array
        Input features (e.g., daily precipitation values or other predictors)
        Shape: (num_days, ...)
    y : np.array
        Target values (e.g., precipitation at future time steps)
    T : int, optional
        Number of past days to include in each input window (default is 7)
    horizon : int, optional
        Prediction horizon: how many days ahead to predict (default is 3)

    Returns
    -------
    X_window : np.array
        Array of shape (num_samples, T, ...) containing T-day input sequences
    y_window : np.array
        Array of shape (num_samples,) containing corresponding targets
    """

    X_window = []
    y_window = []
    num_days = len(X)

    for t in range(T-1, num_days-horizon):
        X_window.append(X[t-T+1 : t+1])
        y_window.append(y[t + horizon])

    return np.array(X_window), np.array(y_window)


class ClimatologyModel:
    """
    Predicts future precipitation using historical mean for that calendar day.
    Baseline model.
    """
    
    def __init__(self):
        """
        Initialize the climatology model.
        """
        self.day_mean = None

    def fit(self, y_train, dates_train):
        """
        Fit the model by computing the mean precipitation for each day-of-year.
        
        Parameters
        ----------
        y_train : np.array
            Training precipitation values
        dates_train : list or array
            Dates corresponding to y_train
        """
        day_of_year = np.array([pd.to_datetime(d).timetuple().tm_yday for d in dates_train])
        unique_days = np.unique(day_of_year)
        self.day_mean = {day: np.mean(y_train[day_of_year == day]) for day in unique_days}

    def predict(self, dates_test):
        """
        Predict precipitation for future dates using historical means.
        
        Parameters
        ----------
        dates_test : list or array
            Dates for which predictions are required
        
        Returns
        -------
        np.array
            Predicted precipitation values
        """
        day_of_year = np.array([pd.to_datetime(d).timetuple().tm_yday for d in dates_test])
        return np.array([self.day_mean[day] for day in day_of_year])


class PersistenceModel:
    """
    Predicts future precipitation as the average of the last W observed days.
    Baseline model.
    """

    def __init__(self, window=7):
        """
        Initialize persistence model.
        
        Parameters
        ----------
        window : int, optional
            Number of past days to average for prediction (default is 7)
        """
        self.window = window
        self.y_train = None

    def fit(self, y_train):
        """
        Store training series for future prediction.
        
        Parameters
        ----------
        y_train : np.array
            Training precipitation values
        """
        self.y_train = y_train

    def predict(self, y_test_window, horizon=1):
        """
        Predict using the persistence model.
        
        Parameters
        ----------
        y_test_window : np.array
            New observations (used to update history)
        horizon : int, optional
            Prediction horizon (default is 1)
        
        Returns
        -------
        np.array
            Predicted precipitation values
        """
        history = list(self.y_train)
        preds = []

        for i in range(len(y_test_window)):
            avg = np.mean(history[-self.window:])
            preds.append(avg)
            history.append(y_test_window[i])

        return np.array(preds)


class LinearRegressionModel:
    """
    Linear regression baseline model for precipitation forecasting.
    Flattens T-day input windows and applies standard linear regression.
    Baseline model.
    """
    
    def __init__(self):
        """
        Initialize linear regression model.
        """
        self.model = LinearRegression()
        self.T = None
        self.d = None

    def fit(self, X_train, y_train):
        """
        Fit the linear regression model.
        
        Parameters
        ----------
        X_train : np.array
            Input features of shape (num_samples, T, d)
        y_train : np.array
            Target values of shape (num_samples,)
        """
        self.T = X_train.shape[1]
        self.d = X_train.shape[2]

        X_flat = X_train.reshape(len(X_train), self.T * self.d)
        self.model.fit(X_flat, y_train)

    def predict(self, X_test):
        """
        Predict using the linear regression model.
        
        Parameters
        ----------
        X_test : np.array
            Input features of shape (num_samples, T, d)
        
        Returns
        -------
        np.array
            Predicted values
        """
        X_flat = X_test.reshape(len(X_test), self.T * self.d)
        return self.model.predict(X_flat)

