import config

import numpy as np
import config

class AnomalyDetector:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.errors = []  # Store (Actual - Forecast) residuals
        
    def detect(self, current_load: float, forecast_load: float) -> str:
        """
        Stateful Anomaly Detection using Statistical Control (Z-Score).
        Rule: If Z-Score > 3 (3-Sigma Event) -> DDoS / SPIKE.
        """
        # Calculate current residual
        error = current_load - forecast_load
        
        # Add to history
        self.errors.append(error)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)
            
        # Need enough data to compute stats
        if len(self.errors) < 10:
            return "NORMAL"
            
        # Compute Stats (Mean & StdDev of Residuals)
        mean_error = np.mean(self.errors)
        std_error = np.std(self.errors)
        
        # Avoid division by zero
        if std_error == 0:
            std_error = 1
            
        # Calculate Z-Score
        z_score = (error - mean_error) / std_error
        
        # ─────────────────────────────────────────────────────────────
        # ANOMALY RULES (Statistical)
        # ─────────────────────────────────────────────────────────────
        
        # Rule 1: DDoS / Huge Spike ( > 3 Sigma )
        if z_score > 3.0:
            return "DDoS / SPIKE DETECTED"
            
        # Rule 2: Sudden Drop ( < -3 Sigma )
        if z_score < -3.0:
            return "SUDDEN DROP DETECTED"
            
        # Rule 3: High Load Warning ( > 2 Sigma )
        if z_score > 2.0:
            return "HIGH LOAD WARNING"
            
        return "NORMAL"
