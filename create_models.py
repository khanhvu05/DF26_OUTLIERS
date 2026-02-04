import numpy as np
import pandas as pd
import pickle
import os
import sys

# Add current directory (src) to sys.path to ensure 'engine' package is found as top-level
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    # Import as 'engine.mocks' so pickle saves it as such (matching app.py context)
    from engine.mocks import MockModel, MockScaler, MockLSTM
except ImportError:
    print("Error: Could not import 'engine.mocks'. Make sure you are in 'src' directory.")
    sys.exit(1)

def generate_dummy_scaler(output_dir):
    """
    Generates a dummy Scaler for testing.
    """
    scaler = None
    try:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit on some random data
        dummy_data = np.array([100, 2000]).reshape(-1, 1) # Min 100, Max 2000
        scaler.fit(dummy_data)
        print("✓ Created real MinMaxScaler")
    except ImportError:
        print("Sklearn not installed. Using MockScaler.")
        scaler = MockScaler()

    os.makedirs(output_dir, exist_ok=True)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler to {scaler_path}")

def generate_dummy_lstm(output_dir):
    """
    Generates a dummy LSTM model for testing.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        
        # 2. Create Dummy Model
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=(30, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Dummy train just to initialize weights properly
        X = np.random.rand(10, 30, 1)
        y = np.random.rand(10, 1)
        model.fit(X, y, epochs=1, verbose=0)
        
        model_path = os.path.join(output_dir, "lstm_model.h5")
        model.save(model_path)
        print(f"✓ Saved dummy LSTM model to {model_path}")
        
    except ImportError as e:
        print("Tensorflow not installed. Creating MockLSTM.")
        # Fallback to MockLSTM
        model = MockLSTM()
        
        model_path = os.path.join(output_dir, "lstm_model.h5")
        # Save as pickle (loader logic handles this now)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved MockLSTM model to {model_path}")
        return

def generate_dummy_sklearn_models(output_dir):
    """
    Generates dummy objects for ARIMA/Prophet.
    """
    os.makedirs(output_dir, exist_ok=True)
            
    # Mock ARIMA
    arima_path = os.path.join(output_dir, "arima_model.pkl")
    with open(arima_path, 'wb') as f:
        pickle.dump(MockModel(), f)
    print(f"✓ Saved dummy ARIMA to {arima_path}")
    
    # Mock Prophet
    prophet_path = os.path.join(output_dir, "prophet_model.pkl")
    with open(prophet_path, 'wb') as f:
        pickle.dump(MockModel(), f)
    print(f"✓ Saved dummy Prophet to {prophet_path}")

if __name__ == "__main__":
    # Create models directory structure
    base_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(base_dir, "DATA", "models")
    
    print("=" * 50)
    print("  GENERATING DUMMY MODELS FOR DEMO (Final Fix)")
    print("=" * 50)
    print(f"Output directory: {models_dir}\n")
    
    generate_dummy_scaler(models_dir)
    generate_dummy_lstm(models_dir)
    generate_dummy_sklearn_models(models_dir)
    
    print("\n" + "=" * 50)
    print("  ✓ ALL MODELS GENERATED SUCCESSFULLY!")
    print("=" * 50)
