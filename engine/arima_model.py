import numpy as np
import pandas as pd
import time
from typing import List, Union, Tuple
import logging

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None

logger = logging.getLogger(__name__)

class CustomARIMAPredictor:
    """
    Ported from '26.02.03_Build_Arima_3.ipynb'.
    Implements ARIMA with Log-Linear Amplitude Correction.
    Trains on-the-fly using the provided history window.
    """
    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2), look_back: int = 1000):
        self.order = order
        self.look_back = look_back # Limit training data size for speed
    
    def build_peak_feature(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        Exogenous feature: 1 = peak hour (9h - 22h), 0 = off-peak.
        """
        # Ensure index is DatetimeIndex
        if not isinstance(index, pd.DatetimeIndex):
             index = pd.to_datetime(index)
        return ((index.hour >= 9) & (index.hour <= 22)).astype(int).reshape(-1, 1)

    def build_seasonal_profiles(self, series_log: pd.Series):
        """
        Builds daily and hourly seasonal profiles from log-transformed series.
        """
        df = series_log.to_frame('y')
        df['minute_of_day'] = df.index.hour * 60 + df.index.minute
        df['minute_of_hour'] = df.index.minute

        S_daily = df.groupby('minute_of_day')['y'].mean()
        S_short = df.groupby('minute_of_hour')['y'].mean()

        return S_daily, S_short

    def remove_seasonality(self, series_log: pd.Series, S_daily, S_short):
        mod = series_log.copy()
        
        # Safe access to seasonal profiles (handle missing keys if any)
        # Using map/reindex is safer but direct access is faster if indices match
        md = mod.index.hour * 60 + mod.index.minute
        mh = mod.index.minute
        
        # Use reindex to align with current index structure if needed, 
        # but here we just map values.
        # Note: S_daily is indexed by minute_of_day (0-1439)
        # S_short is indexed by minute_of_hour (0-59)
        
        daily_pattern = S_daily.reindex(md).values
        short_pattern = S_short.reindex(mh).values
        
        # Fill NaNs if any minute is not in training profile (rare but possible)
        # We assume 0 seasonality if missing
        daily_pattern = np.nan_to_num(daily_pattern)
        short_pattern = np.nan_to_num(short_pattern)

        mod = mod - daily_pattern
        mod = mod - short_pattern

        return mod

    def add_seasonality(self, pred_ds_values: np.ndarray, index: pd.DatetimeIndex, S_daily, S_short):
        if not isinstance(index, pd.DatetimeIndex):
             index = pd.to_datetime(index)
             
        md = index.hour * 60 + index.minute
        mh = index.minute

        daily_pattern = S_daily.reindex(md).values
        short_pattern = S_short.reindex(mh).values
        
        daily_pattern = np.nan_to_num(daily_pattern)
        short_pattern = np.nan_to_num(short_pattern)

        pred_log = pred_ds_values + daily_pattern + short_pattern
        return pred_log

    def predict(self, recent_data: Union[pd.DataFrame, np.ndarray, List[float]], steps: int = 1) -> List[float]:
        """
        Trains ARIMA on `recent_data` and forecasts `steps` ahead.
        Expects `recent_data` to have 'timestamp' and 'requests' (or 'total_bytes') columns.
        If 'timestamp' is missing, it will infer or error.
        """
        if ARIMA is None:
            logger.error("statsmodels is not installed. Returning naive predictions.")
            if isinstance(recent_data, pd.DataFrame) and 'requests' in recent_data.columns:
                 return [float(recent_data.iloc[-1]['requests'])] * steps
            elif isinstance(recent_data, (list, np.ndarray)) and len(recent_data) > 0:
                 return [float(recent_data[-1])] * steps
            else:
                 return [0.0] * steps

        # 1. Prepare Training Data
        if isinstance(recent_data, (list, np.ndarray)):
            df = pd.DataFrame({'requests': recent_data})
            # Fallback: create dummy index
            df['time'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='5T')
            df = df.set_index('time')
        else:
            df = recent_data.copy()
            if 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('time').set_index('time')
            else:
                 # Fallback: create dummy index
                 df.index = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='5T')
        
        target_col = 'requests' if 'requests' in df.columns else df.columns[0]
        train = df[target_col]

        # Limit history
        if len(train) > self.look_back:
            train = train[-self.look_back:]

        # 2. Log Transform
        train_log = np.log1p(train)

        # 3. Seasonality
        S_daily, S_short = self.build_seasonal_profiles(train_log)
        train_ds = self.remove_seasonality(train_log, S_daily, S_short)

        # 4. Scale Residuals
        std = train_ds.std()
        if std == 0: std = 1e-6
        train_ds = train_ds / std

        # 5. Exogenous Features (Peak Hour)
        exog_train = self.build_peak_feature(train_ds.index)
        
        # Prepare Exog for Forecast
        last_time = df.index[-1]
        # Infer frequency
        inferred_freq = pd.infer_freq(df.index)
        if not inferred_freq:
            inferred_freq = '5T' # Default

        future_dates = pd.date_range(start=last_time, periods=steps+1, freq=inferred_freq)[1:]
        exog_test = self.build_peak_feature(future_dates)

        # 6. Train ARIMA
        # Enforce order from config or default
        try:
            model = ARIMA(train_ds, exog=exog_train, order=self.order)
            res = model.fit()
            
            # 7. Forecast
            pred_ds = res.forecast(steps=steps, exog=exog_test)
            
            # 8. Inverse Transform
            # Scale back
            pred_ds = pred_ds * std
            
            # Add seasonality
            pred_log = self.add_seasonality(pred_ds.values, future_dates, S_daily, S_short)
            
            # Exp transform
            pred = np.expm1(pred_log)
            pred = np.maximum(pred, 0)
            
            # 9. Amplitude Correction (Optional/Advanced)
            # Notebook mentions: scale = a * log(y_pred) + b. 
            # But the notebook code assumes `evaluate` calculates error ratios on TEST set.
            # In live inference, we don't have y_true for the future.
            # We can only learn `a` and `b` from TRAINING errors (residuals).
            # For simplicity and robustness in this first integration, we perform the core ARIMA forecast.
            # Use ratio correction if we had a hold-out set, but here we train on all recent data.
            
            return pred.tolist()

        except Exception as e:
            logger.error(f"ARIMA prediction failed: {e}")
            # Fallback
            return [float(train.iloc[-1])] * steps

if __name__ == "__main__":
    # Test block
    dates = pd.date_range(start="2023-01-01", periods=200, freq="5T")
    # Sine wave with trend
    values = [100 + 50*np.sin(i/10) + i*0.5 + np.random.normal(0, 5) for i in range(200)]
    df_test = pd.DataFrame({'timestamp': dates, 'requests': values})
    
    predictor = CustomARIMAPredictor()
    t0 = time.time()
    preds = predictor.predict(df_test, steps=5)
    print(f"Prediction time: {time.time()-t0:.2f}s")
    print("Predictions:", preds)
