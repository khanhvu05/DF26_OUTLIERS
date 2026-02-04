import numpy as np
import pandas as pd

class MockModel:
    """
    Mock model for ARIMA/Prophet that behaves like a trained model.
    """
    def forecast(self, steps):
        # Statsmodels-like
        return np.random.rand(steps) * 100 + 100
        
    def predict(self, n_periods=None, future_df=None):
        if future_df is not None:
            # Prophet-like
            return pd.DataFrame({'yhat': np.random.rand(len(future_df)) * 100 + 100})
        # Sklearn-like or Statsmodels-like
        return np.random.rand(n_periods or 1) * 100 + 100

class MockScaler:
    """
    Mock Scaler that mimics sklearn MinMaxScaler (pass-through).
    """
    def __init__(self):
        self.min_ = 0
        self.scale_ = 1
        
    def fit(self, data):
        pass
        
    def fit_transform(self, data):
        return data
        
    def transform(self, data):
        return data
        
    def inverse_transform(self, data):
        return data

class MockLSTM:
    """
    Mock Keras LSTM model.
    """
    def predict(self, x, verbose=0):
        # x shape: (batch_size, timesteps, features)
        # return shape: (batch_size, units) -> we assume output dim is 1
        batch_size = x.shape[0] if hasattr(x, 'shape') else 1
        return np.random.rand(batch_size, 1)
