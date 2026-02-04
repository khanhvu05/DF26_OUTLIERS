"""
Model Management Module for PLANORA Dashboard
Load, save, and use trained ML models
"""
import pickle
import joblib
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np


class ModelManager:
    """Manage ML models for forecasting"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize ModelManager
        
        Args:
            model_dir: Directory to store/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.model_type = None
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from file
        
        Args:
            model_path: Path to model file (.pkl or .joblib)
            
        Returns:
            bool: Success status
        """
        try:
            file_path = Path(model_path)
            
            if file_path.suffix == '.pkl':
                with open(file_path, 'rb') as f:
                    self.model = pickle.load(f)
            elif file_path.suffix == '.joblib':
                self.model = joblib.load(file_path)
            else:
                st.error(f"Unsupported model format: {file_path.suffix}")
                return False
            
            # Detect model type
            self.model_type = type(self.model).__name__
            st.success(f"✅ Loaded {self.model_type} model from {file_path.name}")
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model, model_name: str, format: str = 'joblib') -> bool:
        """
        Save a trained model to file
        
        Args:
            model: Trained model object
            model_name: Name for the model file
            format: 'pkl' or 'joblib'
            
        Returns:
            bool: Success status
        """
        try:
            file_path = self.model_dir / f"{model_name}.{format}"
            
            if format == 'pkl':
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)
            elif format == 'joblib':
                joblib.dump(model, file_path)
            else:
                st.error(f"Unsupported format: {format}")
                return False
            
            st.success(f"✅ Model saved to {file_path}")
            return True
            
        except Exception as e:
            st.error(f"Error saving model: {e}")
            return False
    
    def predict_multi_horizon(self, features: np.ndarray, horizons: List[int] = [1, 5, 15]) -> Tuple[float, float, float]:
        """
        Make multi-horizon predictions
        
        Args:
            features: Input features for prediction
            horizons: List of forecast horizons (minutes)
            
        Returns:
            Tuple of predictions for each horizon
        """
        if self.model is None:
            st.warning("⚠️ No model loaded. Using simulation mode.")
            # Fallback to simulation
            from utils.ai_models import get_ai_prediction_multi_horizon
            return get_ai_prediction_multi_horizon(features[0] if len(features) > 0 else 80, 0)
        
        try:
            # For XGBoost/sklearn models
            if hasattr(self.model, 'predict'):
                predictions = []
                for horizon in horizons:
                    # Adjust features for different horizons if needed
                    pred = self.model.predict(features.reshape(1, -1))[0]
                    predictions.append(int(pred))
                
                return tuple(predictions)
            else:
                st.error("Model does not have predict method")
                return (0, 0, 0)
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return (0, 0, 0)
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {
                'loaded': False,
                'type': None,
                'parameters': None
            }
        
        info = {
            'loaded': True,
            'type': self.model_type,
            'parameters': None
        }
        
        # Try to get model parameters
        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info


class ModelTrainer:
    """Train new models (placeholder for future implementation)"""
    
    @staticmethod
    def train_xgboost(X_train, y_train, params: Optional[dict] = None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets
            params: XGBoost parameters
            
        Returns:
            Trained model
        """
        try:
            import xgboost as xgb
            
            if params is None:
                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100
                }
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            return model
            
        except ImportError:
            st.error("XGBoost not installed. Run: pip install xgboost")
            return None
        except Exception as e:
            st.error(f"Training error: {e}")
            return None
    
    @staticmethod
    def train_arima(data, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Train ARIMA model
        
        Args:
            data: Time series data
            order: ARIMA order (p, d, q)
            
        Returns:
            Trained model
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            model = ARIMA(data, order=order)
            fitted_model = model.fit()
            
            return fitted_model
            
        except ImportError:
            st.error("statsmodels not installed. Run: pip install statsmodels")
            return None
        except Exception as e:
            st.error(f"Training error: {e}")
            return None
