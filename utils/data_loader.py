"""
Data Loading Module for PLANORA Dashboard
Supports CSV, Database, and API data sources
"""
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


class DataLoader:
    """Handle data loading from various sources"""
    
    @staticmethod
    def load_from_csv(file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with loaded data
            
        Expected columns:
            - timestamp: datetime
            - requests_per_minute: int/float
            - (optional) replicas: int
        """
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            st.error(f"Error loading CSV: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def load_from_uploaded_file(uploaded_file) -> pd.DataFrame:
        """
        Load data from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(uploaded_file)
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict with validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required columns
        required_cols = ['timestamp', 'requests_per_minute']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            validation['is_valid'] = False
            validation['errors'].append(f"Missing required columns: {missing_cols}")
        
        # Check for nulls
        if df.isnull().any().any():
            validation['warnings'].append("Data contains null values")
        
        # Check timestamp ordering
        if 'timestamp' in df.columns:
            if not df['timestamp'].is_monotonic_increasing:
                validation['warnings'].append("Timestamps are not in chronological order")
        
        return validation
    
    @staticmethod
    def generate_sample_data(num_points: int = 100) -> pd.DataFrame:
        """
        Generate sample data for testing
        
        Args:
            num_points: Number of data points to generate
            
        Returns:
            DataFrame with sample data
        """
        import numpy as np
        
        timestamps = [datetime.now() - timedelta(minutes=i) for i in range(num_points, 0, -1)]
        requests = [int(80 + 50 * np.sin(i/5) + np.random.randint(-10, 10)) for i in range(num_points)]
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'requests_per_minute': requests
        })


class DataPreprocessor:
    """Preprocess data for model input"""
    
    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract time-based features from data
        
        Args:
            df: DataFrame with timestamp column
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['minute'] = df['timestamp'].dt.minute
        
        return df
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, target_col: str, lags: list = [1, 5, 15]) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: DataFrame
            target_col: Column to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return df.dropna()
