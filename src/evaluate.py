import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_metrics(y_true, y_pred):
    """Calculates and returns RMSE, MAE, and MAPE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

def print_metrics(metrics, model_name):
    """Prints the evaluation metrics for a model."""
    print(f"\n--- {model_name} Performance ---")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("-------------------------")
