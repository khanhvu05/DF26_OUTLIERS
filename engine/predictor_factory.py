import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from .arima_model import CustomARIMAPredictor

class BasePredictor:
    def predict(self, recent_data: Union[pd.DataFrame, np.ndarray], steps: int = 1) -> List[float]:
        raise NotImplementedError

class NaivePredictor(BasePredictor):
    """
    Predicts the last value repeatedly (flat line).
    Fallback when no model is available.
    """
    def predict(self, recent_data: Union[pd.DataFrame, np.ndarray], steps: int = 1) -> List[float]:
        if isinstance(recent_data, pd.DataFrame):
            last_val = recent_data.iloc[-1]['requests'] if 'requests' in recent_data else 0
        else:
            last_val = recent_data[-1] if len(recent_data) > 0 else 0
            
        return [float(last_val)] * steps

class LSTMPredictor(BasePredictor):
    def __init__(self, model, scaler, look_back: int = 30):
        self.model = model
        self.scaler = scaler
        self.look_back = look_back
        
    def predict(self, recent_data: pd.DataFrame, steps: int = 1) -> List[float]:
        # Implementation assumes usage of 'requests' column
        # Needs exactly `look_back` data points
        if len(recent_data) < self.look_back:
            # Not enough data
            return NaivePredictor().predict(recent_data, steps)
            
        # 1. Prepare Data
        # Extract last `look_back` values
        series = recent_data['requests'].values[-self.look_back:].reshape(-1, 1)
        
        # 2. Scale
        if self.scaler:
            series_scaled = self.scaler.transform(series)
        else:
            series_scaled = series
            
        # 3. Reshape for LSTM (samples, time steps, features)
        X_input = series_scaled.reshape((1, self.look_back, 1))
        
        # 4. Predict
        # Note: This simple loop predicts 1 step at a time if we wanted multi-step
        # But if model output is 1 unit, we need recursive prediction for multi-step
        
        predictions = []
        current_input = X_input
        
        for _ in range(steps):
            y_pred_scaled = self.model.predict(current_input, verbose=0)
            y_val_scaled = y_pred_scaled[0, 0]
            predictions.append(y_val_scaled)
            
            # Update input for next step (sliding window)
            # Remove first element, append new prediction
            new_step = np.array([[[y_val_scaled]]]) 
            current_input = np.append(current_input[:, 1:, :], new_step, axis=1)
            
        # 5. Inverse Scale
        predictions = np.array(predictions).reshape(-1, 1)
        if self.scaler:
            final_preds = self.scaler.inverse_transform(predictions)
        else:
            final_preds = predictions
            
        return final_preds.flatten().tolist()

class ArimaPredictor(BasePredictor):
    def __init__(self, model):
        self.model = model
        
    def predict(self, recent_data: Union[pd.DataFrame, np.ndarray], steps: int = 1) -> List[float]:
        # ARIMA usually needs the full history to refit or update, but for online inference 
        # we might just use the `forecast` method if it's a statsmodels resultwrapper.
        # Ideally, we should update the model with new observations.
        # For this demo, we assume the model object can forecast from the end of training data
        # OR we just simulate a forecast if it's a dummy object.
        
        # In a real scenario, we'd do: model.apply(new_data) -> forecast
        try:
            # Check if it has a forecast method (statsmodels)
            if hasattr(self.model, 'forecast'):
                # This is tricky without refitting. 
                # Simplified: Assume model is static and just gives a trend or we use a simple logic
                # if it's a dummy wrapper.
                return self.model.forecast(steps=steps).tolist() 
            elif hasattr(self.model, 'predict'):
                 return self.model.predict(n_periods=steps).tolist()
        except:
            pass
            
        # Fallback if complex model update is not implemented
        return NaivePredictor().predict(recent_data, steps)

class ProphetPredictor(BasePredictor):
    def __init__(self, model):
        self.model = model
        
    def predict(self, recent_data: Union[pd.DataFrame, np.ndarray], steps: int = 1) -> List[float]:
        # Prophet predicts based on Dataframe with ds (time)
        # We need the timestamps from recent_data
        if not isinstance(recent_data, pd.DataFrame) or 'timestamp' not in recent_data.columns:
             return NaivePredictor().predict(recent_data, steps)
             
        last_time = recent_data['timestamp'].iloc[-1]
        future_dates = [last_time + pd.Timedelta(seconds=60*i) for i in range(1, steps+1)]
        future_df = pd.DataFrame({'ds': future_dates})
        
        try:
            forecast = self.model.predict(future_df)
            return forecast['yhat'].values.tolist()
        except:
            return NaivePredictor().predict(recent_data, steps)

class HybridPredictor(BasePredictor):
    """
    Combines Prophet (Trend) + LSTM (Residual).
    """
    def __init__(self, prophet_predictor, lstm_predictor):
        self.prophet = prophet_predictor
        self.lstm = lstm_predictor
        
    def predict(self, recent_data: Union[pd.DataFrame, np.ndarray], steps: int = 1) -> List[float]:
        # 1. Get Prophet Prediction (Trend)
        trend = self.prophet.predict(recent_data, steps)
        
        # 2. Get LSTM Prediction (Residual/Error)
        # In a real hybrid, LSTM predicts (Actual - Trend).
        # Here we assume the LSTM provided is already trained on residuals.
        residuals = self.lstm.predict(recent_data, steps)
        
        # 3. Combine
        return [t + r for t, r in zip(trend, residuals)]

class PredictorFactory:
    @staticmethod
    def get_predictor(model_type: str, models: Dict[str, any], scaler=None) -> BasePredictor:
        """
        models: Dict containing 'lstm', 'arima', 'prophet'
        """
        mt = model_type.lower()
        
        if mt == 'lstm':
            if models.get('lstm'):
                return LSTMPredictor(models['lstm'], scaler)
        

        elif mt == 'arima':
            # Use the new CustomARIMAPredictor that trains on the fly
            # We don't strictly need a pre-loaded model object from `models` dict
            # because CustomARIMAPredictor initializes its own internal ARIMA.
            # However, we can pass config parameters if they were in `models`.
            return CustomARIMAPredictor()
        
        elif mt == 'prophet':
            if models.get('prophet'):
                return ProphetPredictor(models['prophet'])
        
        elif mt == 'hybrid':
            p_model = models.get('prophet')
            l_model = models.get('lstm')
            # Hybrid needs both, AND the LSTM should ideally be the residual one.
            # For demo, we might reuse the standard LSTM if a specific one isn't available,
            # but usually they are distinct.
            if p_model and l_model:
                return HybridPredictor(
                    ProphetPredictor(p_model),
                    LSTMPredictor(l_model, scaler)
                )
        
        return NaivePredictor()
