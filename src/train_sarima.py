import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
from src.preprocess import load_data, get_train_test_split
from src.evaluate import calculate_metrics, print_metrics
from src.utils import plot_predictions, save_model

def train_sarima(train_data):
    """Trains a SARIMA model using a simple grid search over small parameter set."""
    print("\nTraining SARIMA model (statsmodels)...")
    p = d = q = range(0, 2)  # small grid for demo
    best_aic = float('inf')
    best_order = None
    best_model = None

    for order in product(p, d, q):
        try:
            model = SARIMAX(train_data['close'], order=order, enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best_order = order
                best_model = res
        except Exception:
            continue

    print(f"Best SARIMA order: {best_order}, AIC: {best_aic:.2f}")
    return best_model

if __name__ == '__main__':
    # Load and split data
    df = load_data('data/btc.csv')
    train_df, test_df = get_train_test_split(df)

    # Train the model
    sarima_model = train_sarima(train_df)

    # Forecast for the test period
    predictions = sarima_model.get_forecast(steps=len(test_df))
    pred_series = predictions.predicted_mean
    
    # Create a predictions DataFrame
    pred_df = pd.DataFrame(pred_series.values, index=test_df.index, columns=['predicted_close'])

    # Evaluate the model
    metrics = calculate_metrics(test_df['close'], pred_df['predicted_close'])
    print_metrics(metrics, "SARIMA")

    # Plot the results
    plot_predictions(test_df['close'], pred_df['predicted_close'], "SARIMA")

    # Save the model
    save_model(sarima_model, 'sarima.pkl')
