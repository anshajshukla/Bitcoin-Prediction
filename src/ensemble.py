import pandas as pd
import numpy as np
from src.utils import load_model, plot_predictions
from src.evaluate import calculate_metrics, print_metrics
from src.preprocess import load_data, get_train_test_split
from src.train_lstm import create_sequences
from src.train_xgboost import create_features
from sklearn.preprocessing import MinMaxScaler

def generate_all_predictions(df, train_df, test_df):
    """Load all models and generate predictions for the test set."""
    all_preds = pd.DataFrame(index=test_df.index)

    # --- SARIMA Predictions ---
    print("\nGenerating SARIMA predictions...")
    sarima_model = load_model('sarima.pkl')
    sarima_preds, _ = sarima_model.predict(n_periods=len(test_df), return_conf_int=True)
    all_preds['sarima'] = sarima_preds

    # --- Prophet Predictions ---
    print("Generating Prophet predictions...")
    prophet_model = load_model('prophet.pkl')
    future = prophet_model.make_future_dataframe(periods=len(test_df), freq='H')
    prophet_forecast = prophet_model.predict(future)
    all_preds['prophet'] = prophet_forecast.set_index('ds').loc[test_df.index]['yhat']

    # --- LSTM Predictions ---
    print("Generating LSTM predictions...")
    lstm_model = load_model('lstm.h5')
    sequence_length = 60
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_df['close'].values.reshape(-1, 1))
    total_dataset = pd.concat((train_df['close'], test_df['close']), axis=0)
    inputs = total_dataset[len(total_dataset) - len(test_df) - sequence_length:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    X_test, _ = create_sequences(inputs, sequence_length)
    lstm_preds = lstm_model.predict(X_test)
    lstm_preds = scaler.inverse_transform(lstm_preds)
    all_preds['lstm'] = lstm_preds

    # --- XGBoost Predictions ---
    print("Generating XGBoost predictions...")
    xgboost_model = load_model('xgboost.json')
    test_featured = create_features(test_df)
    features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    X_test_xgb = test_featured[features]
    xgb_preds = xgboost_model.predict(X_test_xgb)
    all_preds['xgboost'] = xgb_preds

    return all_preds

if __name__ == '__main__':
    df = load_data('data/btc.csv')
    train_df, test_df = get_train_test_split(df)

    # Generate predictions from all models
    predictions_df = generate_all_predictions(df, train_df, test_df)

    # Calculate ensemble prediction (simple average)
    predictions_df['ensemble'] = predictions_df.mean(axis=1)

    print("\n--- Individual and Ensemble Model Performance ---")
    for model_name in predictions_df.columns:
        metrics = calculate_metrics(test_df['close'], predictions_df[model_name])
        print_metrics(metrics, model_name.upper())

    # Plot the ensemble predictions
    plot_predictions(test_df['close'], predictions_df['ensemble'], "Ensemble")
