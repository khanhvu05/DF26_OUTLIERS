"""
Autoscaling Logic for Outliers Demo Dashboard
"""
from datetime import datetime


def scaling_logic(forecast_5m, actual_load, session_state, threshold_up, threshold_down, cooldown):
    """
    Logic quyết định scaling dựa trên dự báo 5 phút
    
    Args:
        forecast_5m: Dự báo tải sau 5 phút
        actual_load: Tải hiện tại
        session_state: Streamlit session state
        threshold_up: Ngưỡng scale-out
        threshold_down: Ngưỡng scale-in
        cooldown: Thời gian cooldown (phút)
        
    Returns:
        tuple: (replicas, decision, reason)
    """
    current_time = datetime.now()
    decision = "KEEP"
    reason = "Trong ngưỡng an toàn"
    
    # Kiểm tra Cooldown
    time_since_last_scale = (current_time - session_state.last_scale_time).seconds / 60
    
    if time_since_last_scale > cooldown:
        if forecast_5m > threshold_up:
            session_state.current_replicas += 1
            session_state.last_scale_time = current_time
            decision = "SCALE_UP"
            reason = f"Dự báo tải tăng lên {forecast_5m} req/min"
        elif forecast_5m < threshold_down and session_state.current_replicas > 1:
            session_state.current_replicas -= 1
            session_state.last_scale_time = current_time
            decision = "SCALE_DOWN"
            reason = f"Dự báo tải giảm xuống {forecast_5m} req/min"
    else:
        if forecast_5m > threshold_up or forecast_5m < threshold_down:
            reason = f"Cooldown active ({cooldown}m)"
            
    return session_state.current_replicas, decision, reason


def calculate_cpu_utilization(actual_load, replicas):
    """
    Tính CPU utilization giả lập
    
    Args:
        actual_load: Tải hiện tại
        replicas: Số lượng replicas
        
    Returns:
        float: CPU utilization (0-100%)
    """
    import numpy as np
    cpu_util = (actual_load / replicas) * 0.8 + np.random.randint(-5, 5)
    return min(95, max(20, cpu_util))


def calculate_cost_savings(total_cost_ai, total_cost_fixed):
    """
    Tính phần trăm tiết kiệm chi phí
    
    Args:
        total_cost_ai: Tổng chi phí với AI autoscaling
        total_cost_fixed: Tổng chi phí với cấu hình cố định
        
    Returns:
        float: Phần trăm tiết kiệm
    """
    if total_cost_fixed == 0:
        return 0
    return 100 - (total_cost_ai / total_cost_fixed * 100)
