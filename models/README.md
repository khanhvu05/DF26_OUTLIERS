# Models Directory

This directory contains trained ML models for PLANORA dashboard.

## Supported Formats

- `.joblib` - Scikit-learn/XGBoost models (recommended)
- `.pkl` - Pickle format models

## How to Use

### 1. Train Your Model

```python
import xgboost as xgb
import joblib

# Train model
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/xgboost_model.joblib')
```

### 2. Load in Dashboard

In the sidebar:
1. Select "Production (Real Data)" mode
2. Check "Use Trained Model"
3. Enter path: `models/xgboost_model.joblib`

## Example Models

Place your trained models here:
- `xgboost_model.joblib` - XGBoost forecasting model
- `arima_model.pkl` - ARIMA time series model
- `lstm_model.h5` - LSTM deep learning model (future)

## Model Requirements

Your model should:
- Accept features: `[current_load, hour, day_of_week, lag_1, lag_5, lag_15]`
- Return prediction: single float value (requests/minute)
