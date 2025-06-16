import xgboost as xgb
import pandas as pd
import numpy as np
from src.preprocess import load_data, get_train_test_split
from src.evaluate import calculate_metrics, print_metrics
from src.utils import plot_predictions, save_model

def create_features(df):
    """Create time series features from the index."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

def train_xgboost(train_data):
    """Trains an XGBoost model."""
    print("\nTraining XGBoost model...")
    
    # Create features
    train_featured = create_features(train_data)
    
    features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    target = 'close'

    X_train = train_featured[features]
    y_train = train_featured[target]

    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        objective='reg:squarederror',
        early_stopping_rounds=10
    )

    # Need an eval set for early stopping
    # We'll just use the last part of the training set as a pseudo-validation set
    split_date = train_df.index[-1] - pd.Timedelta(days=1)
    train_split = train_featured.loc[train_featured.index <= split_date]
    val_split = train_featured.loc[train_featured.index > split_date]

    X_train_split, y_train_split = train_split[features], train_split[target]
    X_val_split, y_val_split = val_split[features], val_split[target]

    model.fit(X_train_split, y_train_split,
              eval_set=[(X_val_split, y_val_split)],
              verbose=False)

    return model

if __name__ == '__main__':
    # Load and split data
    df = load_data('data/btc.csv')
    train_df, test_df = get_train_test_split(df)

    # Train the model
    xgboost_model = train_xgboost(train_df)

    # Make predictions
    test_featured = create_features(test_df)
    features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
    X_test = test_featured[features]
    predictions = xgboost_model.predict(X_test)

    # Create a predictions DataFrame
    pred_df = pd.DataFrame(predictions, index=test_df.index, columns=['predicted_close'])

    # Evaluate the model
    metrics = calculate_metrics(test_df['close'], pred_df['predicted_close'])
    print_metrics(metrics, "XGBoost")

    # Plot the results
    plot_predictions(test_df['close'], pred_df['predicted_close'], "XGBoost")

    # Save the model
    xgboost_model.save_model('models/xgboost.json')
    print("Model saved to models/xgboost.json")
