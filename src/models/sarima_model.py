import os
import joblib
import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split


class SarimaForecastModel:
    """
    A forecasting model using auto-SARIMA to find the best model parameters.

    This class wraps `pmdarima.auto_arima` to automatically select the best
    (p, d, q)(P, D, Q, m) parameters, fit the model, and make forecasts.
    It is well-suited for time series with trend and seasonality.
    """

    def __init__(self, horizon: int, seasonal_period: int = 96, **auto_arima_params):
        """
        Initializes the model.

        Args:
            horizon (int): The number of periods to forecast into the future.
            seasonal_period (int): The number of time steps for a single seasonal period (e.g., 96 for daily seasonality with 15-min data).
            **auto_arima_params: Additional parameters passed to `pmdarima.auto_arima`.
        """
        self.horizon = horizon
        self.seasonal_period = seasonal_period
        self.model = None
        self.fitted_ = False
        self.target_col_ = None

        # Default auto_arima parameters, can be overridden by user
        self.auto_arima_params = {
            'start_p': 1,
            'start_q': 1,
            'test': 'adf',       # Use ADF test to find 'd'
            'max_p': 5,          # Increase search range for p
            'max_q': 5,          # Increase search range for q
            'm': self.seasonal_period,
            'seasonal': True,
            'start_P': 1,        # Start search for P from 1
            'max_P': 3,          # Increase search range for P
            'max_Q': 3,          # Increase search range for Q
            'max_D': 2,          # Increase search range for D
            'D': None,           # Let auto_arima find the best D
            'seasonal_test': 'ocsb', # Use OCSB test to determine D
            'trace': True,
            'error_action': 'ignore',
            'suppress_warnings': True,
            'stepwise': False    # Exhaustive search instead of stepwise
        }
        self.auto_arima_params.update(auto_arima_params)

    def fit(self, train_df: pd.DataFrame, target_col: str):
        """
        Finds the best SARIMA model and fits it to the training data.

        Args:
            train_df (pd.DataFrame): The training data with a DatetimeIndex.
            target_col (str): The name of the column to forecast.
        """
        self.target_col_ = target_col
        y_train = train_df[self.target_col_]

        print("Starting auto_arima to find the best model...")
        self.model = pm.auto_arima(y_train, **self.auto_arima_params)
        
        print("Best SARIMA model found and fitted.")
        print(self.model.summary())

        self.fitted_ = True
        return self

    def predict(self) -> pd.Series:
        """
        Creates a forecast for the next `horizon` steps.
        """
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet.")

        forecast, conf_int = self.model.predict(n_periods=self.horizon, return_conf_int=True)
        
        # The forecast object is a pd.Series with the correct index
        return forecast

    def save(self, path: str) -> None:
        """Saves the fitted model to a file."""
        if not self.fitted_:
            raise RuntimeError("Model is not fitted yet. Cannot save.")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Loads a model from a file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model_instance = joblib.load(path)
        print(f"Model loaded from {path}")
        return model_instance 