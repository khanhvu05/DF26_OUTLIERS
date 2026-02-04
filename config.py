"""
Configuration Center for Autoscaling Analysis
"""
import os

# --- AWS Color Palette ---
AWS_SQUID_INK = "#232F3E"
AWS_ORANGE = "#FF9900"
AWS_BLUE = "#146EB4"
AWS_LIGHT_BLUE = "#00A8E1"
AWS_DARK_BG = "#16191f"

# --- Simulation Settings ---
SIMULATION_SPEED_DEFAULT = 0.5  # seconds per tick
SIMULATION_STEPS = 200

# --- Autoscaling Parameters ---
DEFAULT_SCALE_OUT_THRESHOLD = 1000  # req/min per server (per autoscaling.txt line 184)
DEFAULT_SCALE_IN_THRESHOLD = 300    # req/min (30% of capacity)
DEFAULT_COOLDOWN_PERIOD = 3        # minutes (ticks)
MIN_REPLICAS = 1
MAX_REPLICAS = 20
INITIAL_REPLICAS = 5
FIXED_REPLICAS = 10

# --- Safety Buffer ---
SAFETY_BUFFER_PERCENT = 25  # Maintain 25% headroom over forecast (e.g., 200 req → 250 capacity)

# --- Cost Model ---
COST_PER_REPLICA_PER_HOUR = 0.5  # $0.5/hour
COST_PER_REPLICA_PER_TICK = COST_PER_REPLICA_PER_HOUR / 60  # $0.0083/min
DAILY_BUDGET = 100.0 # $100 per day
BUSINESS_HOURS_START = 8 # 8:00 AM
BUSINESS_HOURS_END = 18  # 6:00 PM

# --- Anomaly Detection ---
ANOMALY_SPIKE_MULTIPLIER = 1.5
ANOMALY_DROP_MULTIPLIER = 0.5
ANOMALY_DROP_THRESHOLD = 30

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Fix: models/ not data/models/
