import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict

# Internal Imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    from core.autoscaler import Autoscaler
    from core.anomaly import AnomalyDetector
    from engine.loader import ModelLoader
    from engine.predictor_factory import PredictorFactory
    from utils.simulation import TimeTraveler
except ImportError as e:
    st.error(f"Import Error: {e}. Please run from 'src' directory.")
    st.stop()

# ──────────────────────────────────────────────
# 0.  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Outliers DEMO Autoscaling",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# 1.  CUSTOM CSS (CYBERPUNK THEME)
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* Font Imports */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&display=swap');

:root {
    --bg-deep: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 100%);
    --bg-card: linear-gradient(145deg, #141b2d 0%, #1e2640 100%);
    --bg-card2: rgba(30, 42, 69, 0.6);
    --cyan: #00d9ff;
    --cyan-glow: rgba(0, 217, 255, 0.3);
    --green: #00ff88;
    --green-glow: rgba(0, 255, 136, 0.25);
    --amber: #ffab00;
    --amber-glow: rgba(255, 171, 0, 0.3);
    --red: #ff4757;
    --red-glow: rgba(255, 71, 87, 0.3);
    --purple: #a855f7;
    --purple-glow: rgba(168, 85, 247, 0.3);
    --text-hi: #f0f4f8;
    --text-lo: #8b95b0;
    --border: rgba(100, 130, 180, 0.2);
}

/* App Background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f35 50%, #0f1420 100%);
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-hi);
}

/* Metric Cards - Compact Style */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #141b2d 0%, #1e2640 100%);
    border: 1px solid var(--border);
    padding: 10px 12px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 3px 12px rgba(0, 0, 0, 0.4), 0 0 10px var(--cyan-glow);
}
div[data-testid="stMetric"] label {
    font-family: 'Orbitron', sans-serif;
    color: var(--cyan);
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    text-shadow: 0 0 8px var(--cyan-glow);
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.3rem;
    color: var(--text-hi);
    font-weight: 700;
    line-height: 1.2;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    font-size: 0.7rem;
}

/* Server Grid Cards */
.server-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
    gap: 10px;
    margin-top: 12px;
}
.server-card {
    background: linear-gradient(145deg, rgba(30, 42, 69, 0.4) 0%, rgba(30, 42, 69, 0.6) 100%);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    backdrop-filter: blur(8px);
}
.server-card.active {
    border-color: var(--green);
    box-shadow: 0 0 15px var(--green-glow), inset 0 0 10px rgba(0, 255, 136, 0.05);
    transform: scale(1.02);
}
.server-card.inactive {
    opacity: 0.25;
    border-color: rgba(100, 130, 180, 0.1);
    filter: grayscale(0.8);
}
.cpu-bar {
    height: 6px;
    width: 100%;
    background: rgba(50, 50, 70, 0.6);
    margin-top: 8px;
    border-radius: 3px;
    overflow: hidden;
}
.cpu-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--cyan) 0%, var(--purple) 100%);
    border-radius: 3px;
    box-shadow: 0 0 8px var(--cyan-glow);
    transition: width 0.3s ease;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141b2d 0%, #1a2138 100%);
    border-right: 2px solid var(--border);
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
}

/* Titles */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    background: linear-gradient(135deg, var(--cyan) 0%, var(--purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-shadow: 0 0 20px var(--cyan-glow);
    font-weight: 700;
    letter-spacing: 1px;
}

/* Info/Warning/Error Boxes */
div[data-testid="stInfo"],
div[data-testid="stSuccess"],
div[data-testid="stWarning"],
div[data-testid="stError"] {
    border-radius: 10px;
    border-left-width: 4px;
    backdrop-filter: blur(10px);
}

/* Progress bars */
div[data-testid="stProgress"] > div {
    background: linear-gradient(90deg, var(--green) 0%, var(--amber) 60%, var(--red) 100%);
    box-shadow: 0 0 10px var(--green-glow);
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# 2.  HELPER FUNCTIONS
# ──────────────────────────────────────────────
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_hybrid_data(resolution: str) -> pd.DataFrame:
    """Loads pre-calculated predictions."""
    res_map = {'1m': '1min_request_count', '5m': '5min_request_count', '15m': '15min_request_count'}
    folder_name = res_map.get(resolution)
    if not folder_name: return None
    
    path = os.path.join("models", "result_lstm", folder_name, "LSTM", "predictions.csv")
    possible_roots = ["", "../", "../../"]
    
    for root in possible_roots:
        full_path = os.path.join(root, path)
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.rename(columns={'actual': 'requests', 'predicted': 'forecast'}, inplace=True)
                return df
            except Exception:
                continue
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_prophet_data(resolution: str) -> pd.DataFrame:
    """Loads pre-calculated Prophet predictions."""
    res_map = {'1m': '1min_request_count', '5m': '5min_request_count', '15m': '15min_request_count'}
    folder_name = res_map.get(resolution)
    if not folder_name: return None
    
    path = os.path.join("models", "results_prophet", folder_name, "predictions.csv")
    possible_roots = ["", "../", "../../"]
    
    for root in possible_roots:
        full_path = os.path.join(root, path)
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.rename(columns={'actual': 'requests', 'predicted': 'forecast'}, inplace=True)
                return df
            except Exception:
                continue
    return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_raw_data(resolution: str) -> pd.DataFrame:
    """Loads raw test data."""
    filename = f"test_{resolution.replace('m', 'min')}.csv"
    path = os.path.join("data", filename)
    possible_roots = ["", "../", "../../"]
    for root in possible_roots:
        full_path = os.path.join(root, path)
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                if 'timestamp' not in df.columns:
                    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if 'request_count' in df.columns:
                    df.rename(columns={'request_count': 'requests'}, inplace=True)
                return df
            except Exception:
                continue
    return None

@st.cache_data(ttl=3600)
def load_train_data(resolution: str, last_n: int = 30) -> pd.DataFrame:
    """Loads last N points from train data for warm-start."""
    filename = f"train_{resolution.replace('m', 'min')}.csv"
    path = os.path.join("data", filename)
    possible_roots = ["", "../", "../../"]
    
    for root in possible_roots:
        full_path = os.path.join(root, path)
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                if 'timestamp' not in df.columns:
                    df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                if 'request_count' in df.columns:
                    df.rename(columns={'request_count': 'requests'}, inplace=True)
                return df.tail(last_n).reset_index(drop=True)
            except Exception:
                continue
    return None

@st.cache_data(ttl=3600)
def load_model_predictions(model_name: str, resolution: str, _version: int = 1) -> pd.DataFrame:
    """Loads pre-calculated predictions for any model (LSTM/ARIMA/Prophet)."""
    res_map = {'1m': '1min_request_count', '5m': '5min_request_count', '15m': '15min_request_count'}
    folder_name = res_map.get(resolution)
    if not folder_name: return None
    
    # Path mapping
    if model_name == "LSTM":
        path = os.path.join("models", "results_lstm", folder_name, "LSTM", "predictions.csv")
    elif model_name == "Prophet":
        path = os.path.join("models", "results_prophet", folder_name, "predictions.csv")
    elif model_name == "Hybrid":
        path = os.path.join("models", "results_hybrid", folder_name, "hybrid_predictions.csv")
    elif model_name == "ARIMA":
        # Standardized structure
        path = os.path.join("models", "results_arima", folder_name, "predictions.csv")
    else:
        return None
    
    possible_roots = ["", "../", "../../"]
    
    for root in possible_roots:
        full_path = os.path.join(root, path)
        if os.path.exists(full_path):
            try:
                df = pd.read_csv(full_path)
                # df['timestamp'] conversion moved to after renaming
                
                # Rename columns to standard format
                if 'actual' in df.columns:
                    df.rename(columns={'actual': 'requests'}, inplace=True)
                
                if 'predicted' in df.columns:
                    df.rename(columns={'predicted': 'forecast'}, inplace=True)
                elif 'hybrid_pred' in df.columns:
                    df.rename(columns={'hybrid_pred': 'forecast'}, inplace=True)
                elif 'arima_amplitude_pred' in df.columns:
                    df.rename(columns={'arima_amplitude_pred': 'forecast'}, inplace=True)
                elif 'arima_pred' in df.columns:
                    df.rename(columns={'arima_pred': 'forecast'}, inplace=True)
                elif 'yhat' in df.columns:  # Prophet format
                    df.rename(columns={'ds': 'timestamp', 'y': 'requests', 'yhat': 'forecast'}, inplace=True)
                
                # ARIMA specific timestamp & filtering
                if 'time' in df.columns:
                    df.rename(columns={'time': 'timestamp'}, inplace=True)
                
                # Convert timestamp AFTER renaming
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Filter by frequency for ARIMA if column exists
                if 'freq' in df.columns:
                    target_freq = resolution.replace('m', 'min') # 1m -> 1min
                    df = df[df['freq'] == target_freq]
                
                if 'predicted' in df.columns:
                    df.rename(columns={'predicted': 'forecast'}, inplace=True)
                
                # ---------------------------------------------------------
                # MERGE WITH RAW DATA TO GET total_bytes (For Effective Load)
                # ---------------------------------------------------------
                raw_df = load_raw_data(resolution)
                if raw_df is not None:
                    # Ensure timestamp format matches
                    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
                    
                    # Merge left on timestamp to keep prediction rows
                    if 'total_bytes' in raw_df.columns:
                        # Extract needed columns
                        cols_to_merge = ['timestamp', 'total_bytes']
                        if 'unique_users' in raw_df.columns:
                            cols_to_merge.append('unique_users')
                        if 'static_ratio' in raw_df.columns:
                            cols_to_merge.append('static_ratio')
                            
                        df = pd.merge(df, raw_df[cols_to_merge], on='timestamp', how='left')
                        
                        # Fill NaN
                        if 'total_bytes' in df.columns:
                            df['total_bytes'] = df['total_bytes'].fillna(0)
                        if 'unique_users' in df.columns:
                             # If missing, estimate based on requests (e.g., 5 req/user)
                            df['unique_users'] = df['unique_users'].fillna(df['requests'] / 5)
                        if 'static_ratio' in df.columns:
                            df['static_ratio'] = df['static_ratio'].fillna(0.5) # Default
                    else:
                        df['total_bytes'] = 0 # Default if missing
                else:
                    df['total_bytes'] = 0
                
                return df
            except Exception:
                continue
    return None

def get_workload_status(current_load: int, active_replicas: int, forecast: int):
    """Classifies workload into 4 tiers based on capacity utilization."""
    capacity = active_replicas * config.DEFAULT_SCALE_OUT_THRESHOLD
    if capacity == 0:
        return "CRITICAL", "#ff2e63", "🔥"
    
    utilization = (current_load / capacity) * 100
    
    # Check for spike (actual >> forecast)
    if forecast > 0:
        spike_ratio = current_load / forecast
        if spike_ratio > 1.5:  # 50% higher than forecast
            return "SPIKE", "#ff2e63", "⚡"
    
    # Normal classification
    if utilization < 40:
        return "LOW", "#00ff88", "📉"
    elif utilization < 80:
        return "NORMAL", "#00e5ff", "✅"
    elif utilization < 110:
        return "HIGH", "#ffd60a", "📈"
    else:
        return "CRITICAL", "#ff2e63", "🔥"

def render_server_grid(active_replicas: int, max_replicas: int = 12):
    """Generates HTML for the visual server grid."""
    cards = []
    for i in range(max_replicas):
        is_active = i < active_replicas
        cls = "active" if is_active else "inactive"
        icon = "🟢" if is_active else "⚪"
        # Simulate CPU load for visual effect
        cpu = int(np.random.normal(60, 15)) if is_active else 0
        cpu = max(0, min(100, cpu))
        color = "#39ff14" if cpu < 70 else "#ff3b5c"
        
        # Use on-line string concatenation or simple string to avoid indentation issues with Markdown
        card_html = f"""
<div class="server-card {cls}">
    <div style="font-size:1.2rem;">{icon}</div>
    <div style="font-family:'Share Tech Mono'; font-size:0.8rem; color:#e8edf5;">NODE-{i+1}</div>
    <div class="cpu-bar">
        <div class="cpu-fill" style="width:{cpu}%; background:{color};"></div>
    </div>
    <div style="font-size:0.6rem; color:#6b7a99; margin-top:2px">{cpu}% Load</div>
</div>"""
        cards.append(card_html)
    
    return f'<div class="server-grid">{"".join(cards)}</div>'

# ──────────────────────────────────────────────
# 3.  APP LOGIC
# ──────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.title("⚙️ SYSTEM CONTROL")
    st.markdown("---")
    
    simulation_speed = st.slider("Cycle Duration (s)", 0.1, 2.0, config.SIMULATION_SPEED_DEFAULT, step=0.1)
    resolution = st.selectbox("Resolution", ["1m", "5m", "15m"], index=0)
    
    # Model Selection (All Pre-calculated)
    st.markdown("### 🤖 AI Forecasting Model")
    model_type = st.selectbox(
        "Select Model",
        ["LSTM", "Prophet", "Hybrid", "ARIMA"],
        index=2, # Default to Hybrid (best)
        help="""
        **Hybrid**: LSTM + Prophet (Best Accuracy)
        **LSTM**: Deep Learning (Good for short-term)
        **Prophet**: Facebook's TS Model (Good for seasonality)
        """
    )
    
    # Time Travel Control
    st.markdown("### ⏳ Time Travel")
    # Find min/max time from loaded data if possible, else default
    default_start = pd.to_datetime("2024-01-01 00:00:00").time()
    start_time = st.slider(
        "Start Time",
        min_value=pd.to_datetime("2024-01-01 00:00:00").time(),
        max_value=pd.to_datetime("2024-01-01 23:59:00").time(),
        value=default_start,
        step=pd.Timedelta(minutes=15),
        format="HH:mm"
    )

    is_running = st.checkbox("▶  ACTIVATE SYSTEM", value=True)
    
    st.markdown("---")
    st.markdown("### 🛡️ SAFEGUARDS")
    min_replicas = st.number_input("Min Replicas", 1, 10, config.MIN_REPLICAS)
    max_replicas = st.number_input("Max Replicas", 10, 50, config.MAX_REPLICAS)

# Initialize State
if 'history' not in st.session_state:
    st.session_state.history = {'timestamp': [], 'requests': [], 'replicas': [], 'forecast': []}

# Data Loading (Always Pre-calculated)
if ('resolution' not in st.session_state or 
    st.session_state.resolution != resolution or 
    'model_type' not in st.session_state or 
    st.session_state.model_type != model_type or 
    'start_time' not in st.session_state or
    st.session_state.start_time != start_time or
    'simulator' not in st.session_state):
    
    with st.spinner(f'🔄 Loading {model_type} predictions...'):
        st.session_state.resolution = resolution
        st.session_state.model_type = model_type
        st.session_state.start_time = start_time
        
        # Hybrid only supports 5m/15m -> Fallback to LSTM for 1m
        target_model = model_type
        if model_type == "Hybrid" and resolution == "1m":
            st.warning("⚠️ Hybrid model available for 5m/15m only. Showing LSTM for 1m.")
            target_model = "LSTM"
        
        # Load pre-calculated predictions
        df = load_model_predictions(target_model, resolution)
        
        if df is not None:
            # Apply Time Travel Filter
            # Filter rows where time component >= start_time
            # We assume data covers a 24h period or just filter by time of day regardless of date
            df_filtered = df[df['timestamp'].dt.time >= start_time].reset_index(drop=True)
            
            if df_filtered.empty:
                st.warning(f"⚠️ No data found after {start_time}. Showing all data.")
                df_filtered = df
            else:
                st.toast(f"⏩ Jumped to {start_time}", icon="⏳")
                
            st.toast(f"✅ {model_type} Predictions Loaded: {resolution}", icon="📈")
            df = df_filtered
        else:
            st.warning(f"⚠️ {model_type} predictions not found, using synthetic data")
            dates = pd.date_range(start='2024-01-01', periods=1000, freq=resolution.replace('m', 'min'))
            reqs = 1000 + 500 * np.sin(np.arange(1000)/20) + np.random.normal(0, 50, 1000)
            forecasts = reqs + np.random.normal(0, 30, 1000)
            df = pd.DataFrame({
                'timestamp': dates, 
                'requests': list(map(int, reqs)),
                'forecast': list(map(int, forecasts))
            })
        
        st.session_state.simulator = TimeTraveler(df)
        st.session_state.history = {
            'timestamp': [], 
            'requests': [], 
            'replicas': [], 
            'forecast': []
        }

if 'autoscaler' not in st.session_state:
    st.session_state.autoscaler = Autoscaler(min_servers=min_replicas, max_servers=max_replicas)

if 'anomaly_detector' not in st.session_state:
    st.session_state.anomaly_detector = AnomalyDetector(window_size=30)

# No model loading needed - all predictions are pre-calculated

# Layout
st.markdown("## ⚡ OUTLIERS DEMO - AUTOSCALING INTELLIGENCE")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
placeholder_metrics = st.empty()
placeholder_charts = st.empty()

def update():
    sim = st.session_state.simulator
    hist = st.session_state.history
    
    data = sim.next_tick()
    if not data:
        st.info("Simulation Complete")
        st.stop()
        
    curr_req = data.get('requests', 0)
    curr_bytes = data.get('total_bytes', 0) # New: Get Total Bytes
    curr_time = data.get('timestamp', pd.Timestamp.now())
    
    # Forecast Logic (Smart Look Ahead)
    # 1. Calculate Confidence first (using history up to now)
    sim_history = []
    if len(hist['requests']) > 0:
        for i in range(len(hist['requests'])):
            sim_history.append({
                'requests': hist['requests'][i],
                'forecast': hist['forecast'][i],
                'static_ratio': hist['static_ratio'][i] if 'static_ratio' in hist and i < len(hist['static_ratio']) else 0.5
            })
            
    conf_score, conf_details = st.session_state.autoscaler.calculate_confidence_score(sim_history)
    
    # 2. Determine Buffer based on Confidence
    if conf_score > 0.75:
        buffer_steps = 2
    elif conf_score > 0.5:
        buffer_steps = 3
    else:
        buffer_steps = 5
        
    # 3. Peek Future Forecast
    future_data = sim.peek_future(buffer_steps)
    if future_data:
        # Use the forecast from the future point as our "Target Forecast"
        # This means we are reacting to what the AI predicts will happen in 'buffer_steps' minutes
        target_forecast = future_data.get('forecast', curr_req)
    else:
        # End of data, fallback to current
        target_forecast = data.get('forecast', curr_req)
        
    # Store immediate forecast for chart display (visualization only)
    # The chart always shows "What the AI predicts for next step" vs Actual
    immediate_forecast = data.get('forecast', curr_req)

    # Scaling
    # Get last replicas or initial config
    current_replicas = hist['replicas'][-1] if hist['replicas'] else config.INITIAL_REPLICAS
            
    # Cost tracking
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0.0

    # Calculate! 
    # Note: We pass 'target_forecast' (Look Ahead) to the calculator
    # NEW: Pass accumulated cost and current time
    # NEW 2: Pass DDoS params
    unique_users = data.get('unique_users', 0)
    static_ratio = data.get('static_ratio', 0.5)
    
    # Update sim_history with static_ratio for Z-score calc
    if len(sim_history) > 0 and len(hist['timestamp']) > 0:
        # Backward patch last item? No, sim_history is rebuilt from hist.
        # We need to store static_ratio in history to make it available for z-score
        pass
        
    replicas, reason, cost_tick, details = st.session_state.autoscaler.calculate_replicas(
        curr_req, curr_bytes, target_forecast, current_replicas, sim_history, 
        accumulated_cost=st.session_state.total_cost,
        current_time=curr_time,
        unique_users=unique_users,
        static_ratio=static_ratio,
        resolution=resolution
    )
    
    # Update Total Cost
    st.session_state.total_cost += cost_tick
    
    # Anomaly Detection (Statistical Z-Score)
    # Compare current request with immediate forecast for anomaly detection
    anomaly = st.session_state.anomaly_detector.detect(curr_req, immediate_forecast)
    
    # Update History
    # Update History
    hist['timestamp'].append(curr_time)
    hist['requests'].append(curr_req)
    hist['replicas'].append(replicas)
    hist['forecast'].append(immediate_forecast)
    # Store auxiliary for next history build (optional, or just rely on Autoscaler keeping track? Autoscaler is stateless per call here)
    # Autoscaler uses 'recent_history' passed in. We construct 'sim_history' from 'hist'.
    # We should add static_ratio to hist to support history lookup!
    if 'static_ratio' not in hist: hist['static_ratio'] = []
    hist['static_ratio'].append(static_ratio)
    
    # Store Risk Metrics
    if 'effective_load' not in hist: hist['effective_load'] = []
    hist['effective_load'].append(details.get('effective_load', curr_req))
    
    if 'upr' not in hist: hist['upr'] = []
    hist['upr'].append(details.get('upr', immediate_forecast))
    
    # Capacity Visualization - Use UPR from Autoscaler
    # Display the actual target capacity the autoscaler is using
    # UPR already contains Safe_Forecast + Sigma margin
    
    # The capacity line should show what the system is preparing for (UPR)
    # not the raw replica capacity which may be over-provisioned
    current_capacity = details.get('upr', immediate_forecast)
    
    if 'capacity' not in hist: hist['capacity'] = []
    hist['capacity'].append(current_capacity)
    
    # Calculate minutes_per_tick for later use (forecast ahead line)
    minutes_per_tick = {'1m': 1, '5m': 5, '15m': 15}.get(resolution, 1)
    
    if len(hist['timestamp']) > 60:
        for k in hist: hist[k] = hist[k][-60:]
        
    # ═══════════════════════════════════════════════════════════
    # PROFESSIONAL 2-COLUMN LAYOUT (Sidebar + Chart)
    # ═══════════════════════════════════════════════════════════
    
    # Create 2-column layout: Compact metrics sidebar (25%) + Large chart (75%)
    sidebar_col, chart_col = st.columns([1, 3])
    
    with sidebar_col:
        st.markdown("#### 📊 **LIVE METRICS**")
        
        # Traffic
        eff_load = int(details.get('effective_load', curr_req))
        st.metric("🌊 Traffic", f"{int(curr_req):,}", f"{(curr_req - hist['requests'][-2]) if len(hist['requests']) > 1 else 0:+.0f}")
        st.caption(f"Effective: {eff_load:,}")
        
        # Forecast & UPR
        upr_val = int(details.get('upr', immediate_forecast))
        st.metric("🔮 Forecast", f"{int(immediate_forecast):,}")
        st.caption(f"UPR: {upr_val:,}")
        
        # Risk Level
        risk_level = details.get('risk_level', 'NORMAL')
        conf_score = int(details.get('confidence_score', 0)*100)
        risk_color = {"LOW": "🟢", "NORMAL": "🟡", "HIGH": "🟠", "SPIKE": "🔴"}.get(risk_level, "⚪")
        st.metric(f"{risk_color} Risk", risk_level)
        st.caption(f"Confidence: {conf_score}%")
        
        # Nodes & Cost
        delta_replicas = replicas - current_replicas
        st.metric("🖥️ Nodes", f"{replicas}", f"{delta_replicas:+d}" if delta_replicas != 0 else None)
        st.caption(f"Cost: ${st.session_state.total_cost:.1f}")
        
        # Workload State
        workload_type = details.get('workload_type', 'NORMAL')
        w_icon = {"FLASH_CROWD": "⚡", "DDOS": "🛡️", "ANOMALY": "⚠️"}.get(workload_type, "✅")
        st.metric(f"{w_icon} State", workload_type)
        st.caption(f"{details.get('action', 'STABLE')}")
        
        st.markdown("---")
        
        # Strategy Info (Compact)
        strat = details.get('strategy', 'Unknown')
        st.caption(f"**Strategy:** {strat}")
        wp = details.get('w_p', 0.5)
        wr = details.get('w_r', 0.5)
        st.caption(f"**Weights:** UPR {wp:.1f} | Load {wr:.1f}")
        
    with chart_col:
        # Row 1: Main Chart
        fig = go.Figure()
        
        # 1. Capacity Area (Green Zone) - What system can handle
        fig.add_trace(go.Scatter(
            x=hist['timestamp'], 
            y=hist['capacity'], 
            name='⚡ Capacity', 
            line=dict(color='#00ff88', width=2.5, dash='dot'),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)',
            hovertemplate='<b>Capacity</b>: %{y:,.0f}<extra></extra>'
        ))

        # 2. AI Forecast (Blue Line)
        fig.add_trace(go.Scatter(
            x=hist['timestamp'],  
            y=hist['forecast'], 
            name='🔮 AI Forecast', 
            line=dict(color='#00ccff', width=2.5),
            hovertemplate='<b>Forecast</b>: %{y:,.0f}<extra></extra>'
        ))
        
        # 3. Actual Traffic (White Line)
        fig.add_trace(go.Scatter(
            x=hist['timestamp'], 
            y=hist['requests'], 
            name='📊 Actual Traffic', 
            line=dict(color='#ffffff', width=3),
            hovertemplate='<b>Actual</b>: %{y:,.0f}<extra></extra>'
        ))
        
        # Look Ahead Line - Show forecast projection as line instead of marker
        next_row = sim.peek_future(1)
        if next_row:
            next_time = hist['timestamp'][-1] + pd.Timedelta(minutes=minutes_per_tick)
            next_forecast = next_row.get('forecast', 0)
            
            # Draw connecting line from current to next forecast
            fig.add_trace(go.Scatter(
                x=[hist['timestamp'][-1], next_time],
                y=[hist['forecast'][-1], next_forecast],
                name='Forecast Ahead',
                line=dict(color='#00ccff', width=3, dash='solid'),
                mode='lines',
                showlegend=False,
                hovertemplate='<b>Next Forecast</b>: %{y:,.0f}<extra></extra>'
            ))
            
        fig.update_layout(
            title=dict(
                text="<b>📊 REAL-TIME TRAFFIC ANALYSIS - RISK-AWARE AUTOSCALING</b>",
                font=dict(size=18, color='#00d9ff', family='Orbitron'),
                x=0.5,
                xanchor='center'
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(20, 27, 45, 0.6)',
            height=700,
            margin=dict(l=40, r=40, t=60, b=40),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                bgcolor='rgba(30, 42, 69, 0.8)',
                bordercolor='rgba(0, 217, 255, 0.3)',
                borderwidth=1,
                font=dict(size=11, color='#f0f4f8')
            ),
            xaxis=dict(
                gridcolor='rgba(100, 130, 180, 0.1)',
                title=dict(text='Time', font=dict(size=12, color='#8b95b0')),
                tickfont=dict(size=10, color='#8b95b0')
            ),
            yaxis=dict(
                gridcolor='rgba(100, 130, 180, 0.1)',
                title=dict(text='Requests', font=dict(size=12, color='#8b95b0')),
                tickfont=dict(size=10, color='#8b95b0')
            ),
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Row 2: Server Grid + Residuals
        c_grid, c_res = st.columns([1, 1])
        with c_grid:
            st.markdown("### 🖥️ SERVER FLEET STATUS")
            st.markdown(render_server_grid(replicas, max_replicas=12), unsafe_allow_html=True)
            
        with c_res:
            res_val = np.array(hist['requests'], dtype=float) - np.array(hist['forecast'], dtype=float)
            # Color bars based on positive/negative residuals
            colors = ['#00ff88' if x < 0 else '#ff4757' for x in res_val]
            fig2 = go.Figure(go.Bar(
                x=hist['timestamp'], 
                y=res_val, 
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                hovertemplate='<b>Error</b>: %{y:,.0f}<br>Time: %{x}<extra></extra>'
            ))
            fig2.update_layout(
                title=dict(
                    text="<b>⚠️ FORECAST ERROR RESIDUALS</b>",
                    font=dict(size=16, color='#ffab00', family='Orbitron')
                ),
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(20, 27, 45, 0.6)',
                height=250,
                margin=dict(l=40, r=40, t=50, b=30),
                xaxis=dict(
                    gridcolor='rgba(100, 130, 180, 0.1)',
                    tickfont=dict(size=9, color='#8b95b0')
                ),
                yaxis=dict(
                    gridcolor='rgba(100, 130, 180, 0.1)',
                    title=dict(text='Error', font=dict(size=11, color='#8b95b0')),
                    tickfont=dict(size=9, color='#8b95b0'),
                    zeroline=True,
                    zerolinecolor='rgba(255, 255, 255, 0.3)',
                    zerolinewidth=2
                )
            )
            st.plotly_chart(fig2, width='stretch')

if is_running:
    update()
    time.sleep(simulation_speed)
    st.rerun()
else:
    st.info("SYSTEM PAUSED")
