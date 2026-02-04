import math
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import config

class Autoscaler:
    def __init__(self, min_servers=config.MIN_REPLICAS, max_servers=config.MAX_REPLICAS):
        self.cooldown_counter = 0
        self.last_action = "NONE"
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.capacity_per_replica = config.DEFAULT_SCALE_OUT_THRESHOLD
        
    def calculate_confidence_score(self, recent_history: List[Dict]) -> Tuple[float, Dict]:
        """
        Calculates Confidence Score (0.0 - 1.0) based on:
        1. Accuracy (1 - MAPE of last 5 points) - Weight 40%
        2. Data Relevance (Simulated: Time of Day match?) - Weight 30%
        3. Stability (Std Dev of error) - Weight 30%
        """
        if not recent_history or len(recent_history) < 5:
            # Default to high confidence start to avoid panic
            return 0.85, {"accuracy": 0.85, "relevance": 0.9, "stability": 0.8}
            
        # Extract last 5 points
        recent = recent_history[-5:]
        
        # 1. Accuracy (MAPE)
        ape_sum = 0
        valid_points = 0
        for pt in recent:
            actual = pt.get('requests', 0)
            forecast = pt.get('forecast', 0)
            if actual > 0:
                ape = abs(actual - forecast) / actual
                ape_sum += min(ape, 1.0) # Cap error at 100%
                valid_points += 1
        
        mape = ape_sum / valid_points if valid_points > 0 else 0
        accuracy_score = max(0, 1.0 - mape)
        
        # 2. Stability (Inverse of relative std dev of error)
        errors = [abs(pt.get('requests', 0) - pt.get('forecast', 0)) for pt in recent]
        mean_error = np.mean(errors) if errors else 0
        if mean_error > 0:
            cv = np.std(errors) / mean_error
            stability_score = max(0, 1.0 - min(cv, 1.0))
        else:
            stability_score = 1.0
            
        # 3. Relevance (Simulated for this demo)
        # In real system, this compares training distribution vs current distribution
        # For now, we keep it high unless accuracy drops significantly
        relevance_score = 0.9 if accuracy_score > 0.6 else 0.5
        
        # Weighted Final Score
        # Formula: 40% Accuracy + 30% Relevance + 30% Stability
        final_score = (0.4 * accuracy_score) + (0.3 * relevance_score) + (0.3 * stability_score)
        
        return round(final_score, 2), {
            "accuracy": round(accuracy_score, 2),
            "relevance": round(relevance_score, 2),
            "stability": round(stability_score, 2)
        }

    def calculate_effective_load(self, requests: float, bytes_transfer: float) -> float:
        """
        Calculates Effective Load = Requests + Alpha * Bytes.
        Alpha is approx 1 request per 100KB (0.00001).
        """
        ALPHA = 0.00001 # 100KB ~ 1 Request
        return requests + (ALPHA * bytes_transfer)

    def _calculate_upr(self, mean_forecast: float, history_residuals: List[float], confidence: float = 0.90) -> Tuple[float, float]:
        """
        Calculates Upper Prediction Range (UPR) with safety buffer.
        Formula: UPR = (Forecast × 1.25) + Z × Sigma
        The 25% buffer ensures baseline safety before adding uncertainty margin.
        """
        import sys
        sys.path.insert(0, 'D:\\Study\\Year4\\ki2\\AUTOSCALING ANALYSIS\\src')
        import config
        
        if not history_residuals or len(history_residuals) < 5:
            # Fallback if no history: add 20% margin
            sigma = mean_forecast * 0.2
        else:
            sigma = np.std(history_residuals)
            
        # Apply safety buffer to forecast first (create safe baseline)
        buffer_multiplier = 1 + (config.SAFETY_BUFFER_PERCENT / 100)  # 1.25
        safe_forecast = mean_forecast * buffer_multiplier
        
        # Z-score for 90% confidence ~ 1.28
        z_score = 1.28 
        upr = safe_forecast + (z_score * sigma)
        return upr, sigma

    def _classify_risk(self, upr: float, current_load: float, history_loads: List[float]) -> str:
        """
        Classifies risk into LOW, NORMAL, HIGH, SPIKE based on historical quartiles.
        """
        if not history_loads or len(history_loads) < 20:
            return "NORMAL"
            
        p30 = np.percentile(history_loads, 30)
        p70 = np.percentile(history_loads, 70)
        p90 = np.percentile(history_loads, 90)
        
        if upr >= p90: return "SPIKE"
        if upr >= p70: return "HIGH"
        if upr < p30: return "LOW"
        return "NORMAL"

    def _check_prewarm_signal(self, risk_level: str, confidence_score: float, current_time: pd.Timestamp) -> Tuple[bool, int, str]:
        """
        Determines if we need to pre-warm servers.
        Returns: (Signal, Prewarm Minutes, Reason)
        """
        if risk_level not in ["HIGH", "SPIKE"]:
            return False, 0, ""
            
        if confidence_score < 0.5:
            return False, 0, ""
            
        # Strategy from document
        if confidence_score > 0.75:
            return True, 2, "Pre-warm(2m):HighConf"
        elif confidence_score >= 0.5:
            return True, 3, "Pre-warm(3m):MedConf"
            
        return False, 0, ""

    def _calculate_z_score(self, current_val: float, history_vals: List[float]) -> float:
        """Helper to calculate Z-Score against recent history."""
        if not history_vals or len(history_vals) < 5:
            return 0.0
        mean = np.mean(history_vals)
        std = np.std(history_vals)
        if std == 0:
            return 0.0
        return (current_val - mean) / std

    def detect_workload_type(self, current_req: float, forecast_req: float, 
                           current_bytes: float, unique_users: float, static_ratio: float,
                           recent_history: List[Dict]) -> Tuple[str, Dict]:
        """
        Classifies workload: NORMAL, FLASH_CROWD, or DDOS using Fuzzy Logic approach.
        """
        details = {}
        
        # 1. Residual Z-Score
        residual = current_req - forecast_req
        residuals = []
        static_ratios = []
        
        for pt in recent_history[-30:]: 
            r = pt.get('requests', 0) - pt.get('forecast', 0)
            residuals.append(r)
            static_ratios.append(pt.get('static_ratio', 0.5))
            
        residual_z = self._calculate_z_score(residual, residuals)
        details['residual_z'] = round(residual_z, 2)
        
        # 2. Analyze Characteristics
        req_per_user = current_req / unique_users if unique_users > 0 else 0
        bytes_per_req = current_bytes / current_req if current_req > 0 else 0
        
        mean_static = np.mean(static_ratios) if static_ratios else 0.5
        static_dev = abs(static_ratio - mean_static)
        
        details.update({
            'req_per_user': round(req_per_user, 2),
            'bytes_per_req': round(bytes_per_req, 0),
            'static_dev': round(static_dev, 3)
        })
        
        # Fuzzy Scores
        score_high_rpu = min(max((req_per_user - 20) / 50, 0), 1) 
        score_low_bpr = 1.0 if bytes_per_req < 1000 else max(0, 1 - (bytes_per_req / 5000))
        score_high_static_dev = min(static_dev / 0.3, 1.0)
        
        ddos_score = (0.5 * score_high_rpu) + (0.3 * score_low_bpr) + (0.2 * score_high_static_dev)
        score_normal_rpu = 1.0 - score_high_rpu
        score_normal_bpr = 1.0 - score_low_bpr
        flash_score = (0.6 * score_normal_rpu) + (0.4 * score_normal_bpr)
        
        details['ddos_score'] = round(ddos_score, 2)
        details['flash_score'] = round(flash_score, 2)
        
        if ddos_score > 0.6 and ddos_score > flash_score:
            return "DDOS", details
        elif flash_score > 0.5 and residual_z > 2: # Only flash crowd if traffic actually spikes
            return "FLASH_CROWD", details
        elif residual_z > 2:
             # High Z-score but inconclusive signature -> Treat as Anomaly (Safety: Scale)
            # Default to Flash Crowd behavior (allow scaling) but flag it
            return "FLASH_CROWD", details 
        else:
            return "NORMAL", details

    def calculate_replicas(self, current_req: float, current_bytes: float, forecast_req: float, current_replicas: int, recent_history: List[Dict], accumulated_cost: float = 0.0, current_time: pd.Timestamp = None, unique_users: float = 0, static_ratio: float = 0.5, resolution: str = "1m") -> Tuple[int, str, float, dict]:
        """
        Calculates required replicas using Risk-Aware Decision Fusion Strategy.
        """
        reason = []
        details = {}
        
        # A. Constants & Config
        minutes_per_tick = {'1m': 1, '5m': 5, '15m': 15}.get(resolution, 1)
        # Capacity scales with time window
        adjusted_capacity = self.capacity_per_replica * minutes_per_tick 
        
        # B. Detect Workload Type
        workload_type, wl_details = self.detect_workload_type(
            current_req, forecast_req, current_bytes, unique_users, static_ratio, recent_history
        )
        details.update(wl_details)
        details['workload_type'] = workload_type
        
        # C. Calculate Metrics (Effective Load, UPR, Confidence)
        conf_score, conf_details = self.calculate_confidence_score(recent_history)
        
        # Calculate Alpha based on resolution: Alpha = 1 / (resolution * 60 * 10^6)
        alpha = 1.0 / (minutes_per_tick * 60 * 1e6)
        effective_load = current_req + (alpha * current_bytes)
        
        # Collect history for stats
        history_residuals = []
        history_loads = []
        for pt in recent_history:
            # Re-calculate effective load for history
            req = pt.get('requests', 0)
            # Estimate bytes if not stored (fallback)
            byt = pt.get('total_bytes', req * 10000) 
            eff = self.calculate_effective_load(req, byt)
            history_loads.append(eff)
            history_residuals.append(req - pt.get('forecast', 0)) # Approx residual
            
        upr, sigma = self._calculate_upr(forecast_req, history_residuals)
        risk_level = self._classify_risk(upr, effective_load, history_loads)
        
        details.update(conf_details)
        details['confidence_score'] = conf_score
        details['effective_load'] = int(effective_load)
        details['upr'] = int(upr)
        details['risk_level'] = risk_level
        details['sigma'] = round(sigma, 2)
        
        # D. Pre-warm Check
        need_prewarm, prewarm_mins, prewarm_msg = self._check_prewarm_signal(risk_level, conf_score, current_time)
        if need_prewarm:
            reason.append(prewarm_msg)
            
        # ─────────────────────────────────────────────────────────────
        # DECISION FUSION (Formula 3.4.4)
        # ─────────────────────────────────────────────────────────────
        
        # Determine weights based on confidence
        if conf_score > 0.75:
            w_p, w_r = 0.8, 0.2
            details['strategy'] = "Predictive-Driven"
        elif conf_score >= 0.5:
            w_p, w_r = 0.5, 0.5
            details['strategy'] = "Hybrid-Balanced"
        else:
            w_p, w_r = 0.2, 0.8
            details['strategy'] = "Reactive-Driven"
            
        # DDoS Override
        if workload_type == "DDOS":
            target_replicas = current_replicas # Block scaling
            reason.append("🛡️DDoS:Block")
            details['mode'] = "DDOS_BLOCK"
        else:
            # Calculate Fusion Target
            # UPR already contains 25% safety buffer: (Forecast×1.25) + 1.28σ
            # No need for additional multiplication
            
            # Using UPR (Risk Aware) instead of just Mean Forecast
            weighted_load = (w_p * upr) + (w_r * effective_load)
            
            # Pre-warm boost: if pre-warm needed, ensure we use at least UPR-based capacity
            if need_prewarm:
                 weighted_load = max(weighted_load, upr)
            
            target_replicas = math.ceil(weighted_load / adjusted_capacity)
            
            details['weighted_load'] = int(weighted_load)
            details['w_p'] = w_p
            details['w_r'] = w_r
            details['mode'] = "FUSION"

        # ─────────────────────────────────────────────────────────────
        # SAFEGUARD LAYER
        # ─────────────────────────────────────────────────────────────
        is_budget_exceeded = accumulated_cost >= config.DAILY_BUDGET
        if is_budget_exceeded and workload_type != "FLASH_CROWD":
            target_replicas = self.min_servers
            reason.append("💰BudgetLimit")
            
        target_replicas = max(target_replicas, self.min_servers)
        target_replicas = min(target_replicas, self.max_servers)
        
        # Stability / Cooldown
        action = "STABLE"
        
        # Check current replicas (ensure it's not None/0)
        current_replicas = max(current_replicas, 1)

        if target_replicas > current_replicas:
            self.cooldown_counter = config.DEFAULT_COOLDOWN_PERIOD 
            action = "SCALE OUT"
            if workload_type == "FLASH_CROWD": reason.append("⚡FlashCrowd")
            elif risk_level == "SPIKE": reason.append("📈SpikeRisk")
            
        elif target_replicas < current_replicas:
            if self.cooldown_counter > 0:
                target_replicas = current_replicas 
                self.cooldown_counter -= 1
                reason.append(f"Cooldown:{self.cooldown_counter}")
                action = "COOLDOWN"
            else:
                self.cooldown_counter = config.DEFAULT_COOLDOWN_PERIOD
                action = "SCALE IN"
        else:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
        
        final_reason = f"[{action}] " + " | ".join(reason)
        cost = target_replicas * config.COST_PER_REPLICA_PER_TICK * minutes_per_tick
        
        details.update({
            "final_target": target_replicas,
            "action": action,
            "cooldown": self.cooldown_counter
        })
        
        return target_replicas, final_reason, cost, details
