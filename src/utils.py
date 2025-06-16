import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model as load_keras_model
import xgboost as xgb
import os

PLOTS_DIR = 'plots'
MODELS_DIR = 'models'

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def plot_predictions(y_true, y_pred, model_name):
    """Plots actual vs. predicted values."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Actual')
    plt.plot(y_true.index, y_pred, label='Predicted', linestyle='--')
    plt.title(f'{model_name} - Actual vs. Predicted Bitcoin Prices')
    plt.xlabel('Timestamp')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f'{model_name}_predictions.png'))
    plt.show()

def save_model(model, filename):
    """Saves a trained model to a file."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(filename):
    """Loads a trained model from a file."""
    path = os.path.join(MODELS_DIR, filename)
    if path.endswith('.h5'):
        model = load_keras_model(path)
    elif path.endswith('.json'):
        model = xgb.XGBRegressor()
        model.load_model(path)
    else:
        model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model
