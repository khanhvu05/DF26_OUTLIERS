"""
AI Models and Prediction Logic for PLANORA Dashboard
"""
import numpy as np


def get_ai_prediction_multi_horizon(current_load, iteration):
    """
    Dự báo đa thời điểm: 1m, 5m, 15m
    
    Args:
        current_load: Tải hiện tại (requests/minute)
        iteration: Iteration hiện tại trong simulation
        
    Returns:
        tuple: (forecast_1m, forecast_5m, forecast_15m)
        
    Note:
        CHỖ TRỐNG ĐỂ LẮP MODEL THẬT:
        predictions = model.predict(features, horizons=[1,5,15])
    """
    # Giả lập: Dự báo xa hơn = nhiễu lớn hơn
    base_trend = 50 * np.sin((iteration + 1) / 5)  # Xu hướng tương lai
    
    forecast_1m = int(current_load + base_trend * 0.2 + np.random.randint(-10, 15))
    forecast_5m = int(current_load + base_trend * 0.6 + np.random.randint(-20, 30))
    forecast_15m = int(current_load + base_trend * 1.0 + np.random.randint(-30, 50))
    
    return forecast_1m, forecast_5m, forecast_15m


def detect_anomaly(actual_load, forecast_1m, spike_multiplier=1.5, drop_multiplier=0.5, drop_threshold=30):
    """
    Phát hiện bất thường (DDoS, spike, unusual drop)
    
    Args:
        actual_load: Tải thực tế
        forecast_1m: Dự báo 1 phút
        spike_multiplier: Ngưỡng nhân cho spike detection
        drop_multiplier: Ngưỡng nhân cho drop detection
        drop_threshold: Ngưỡng tuyệt đối cho drop
        
    Returns:
        tuple: (is_anomaly, anomaly_message)
    """
    # Nếu tải thực tế vượt quá dự báo 1 phút > 50%, coi là bất thường
    if actual_load > forecast_1m * spike_multiplier:
        return True, "⚠️ Possible DDoS Attack"
    elif actual_load < forecast_1m * drop_multiplier and actual_load < drop_threshold:
        return True, "⚠️ Unusual Drop"
    
    return False, "✅ Normal"


def generate_simulated_load(iteration):
    """
    Tạo tải giả lập theo hàm sin + nhiễu
    
    Args:
        iteration: Iteration hiện tại
        
    Returns:
        int: Tải giả lập (requests/minute)
    """
    return int(80 + 50 * np.sin(iteration / 5) + np.random.randint(-10, 10))
