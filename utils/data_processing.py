import pandas as pd
import os

def load_traffic_data(filepath: str) -> pd.DataFrame:
    """
    Loads traffic data from CSV.
    Expected columns: timestamp, requests, bytes (optional)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
    else:
        # Fallback if no timestamp, generate synthetic?
        # For now assume it exists
        pass
        
    return df

def resample_data(df: pd.DataFrame, rule: str = '1T') -> pd.DataFrame:
    """
    Resample data to a specific time frequency.
    Aggregates: sum of requests, mean of bytes (if exists)
    """
    df_resampled = df.set_index('timestamp').resample(rule).agg({
        'requests': 'sum',
        'bytes': 'mean' # Example
    }).dropna().reset_index()
    
    return df_resampled
