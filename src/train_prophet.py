import pandas as pd
from prophet import Prophet
from src.preprocess import load_data, get_train_test_split
from src.evaluate import calculate_metrics, print_metrics
from src.utils import plot_predictions, save_model

def train_prophet(train_data):
    """Trains a Prophet model."""
    print("\nTraining Prophet model...")
    # Prophet requires columns to be named 'ds' and 'y'
    prophet_df = train_data.reset_index().rename(columns={'timestamp': 'ds', 'close': 'y'})

    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    return model

if __name__ == '__main__':
    # Load and split data
    df = load_data('data/btc.csv')
    train_df, test_df = get_train_test_split(df)

    # Train the model
    prophet_model = train_prophet(train_df)

    # Create a future dataframe for predictions
    future = prophet_model.make_future_dataframe(periods=len(test_df), freq='H')
    forecast = prophet_model.predict(future)

    # Extract predictions for the test set period
    pred_df = forecast.set_index('ds').loc[test_df.index]

    # Evaluate the model
    metrics = calculate_metrics(test_df['close'], pred_df['yhat'])
    print_metrics(metrics, "Prophet")

    # Plot the results
    plot_predictions(test_df['close'], pred_df['yhat'], "Prophet")

    # Plot forecast components
    fig = prophet_model.plot_components(forecast)
    fig.savefig('plots/prophet_components.png')

    # Save the model
    # Note: Prophet models have a custom saving method, but for consistency, 
    # we can wrap it or use a standard format if needed. Here we use joblib.
    save_model(prophet_model, 'prophet.pkl')
