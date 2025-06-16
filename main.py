import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run_script(script_name):
    """Runs a python script in the src directory."""
    script_path = os.path.join('src', script_name)
    print(f"\n{'='*20} Running {script_name} {'='*20}")
    subprocess.run(['python', script_path], check=True)
    print(f"{'='*20} Finished {script_name} {'='*20}")

def main():
    """Main function to run the entire forecasting pipeline."""
    # List of scripts to run in order
    scripts_to_run = [
        'train_sarima.py',
        'train_prophet.py',
        'train_lstm.py',
        'train_xgboost.py',
        'ensemble.py'
    ]

    for script in scripts_to_run:
        try:
            run_script(script)
        except subprocess.CalledProcessError as e:
            print(f"Error running {script}: {e}")
            break
        except FileNotFoundError:
            print(f"Error: {script} not found. Make sure you are in the project root directory.")
            break

def load_bitcoin_data():
    """
    Load Bitcoin price data from CSV file
    Returns: DataFrame with date and price columns
    """
    try:
        # Read the CSV file
        df = pd.read_csv('data/btc.csv')
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("Error: Bitcoin data file not found!")
        return None

def prepare_data(df):
    """
    Prepare data for training
    Input: DataFrame with date and price columns
    Returns: X (features) and y (target) for training
    """
    # Create features based on past prices
    df['price_lag1'] = df['price'].shift(1)  # Previous day's price
    df['price_lag2'] = df['price'].shift(2)  # Price from 2 days ago
    df['price_lag3'] = df['price'].shift(3)  # Price from 3 days ago
    
    # Drop rows with missing values
    df = df.dropna()
    
    # Prepare features (X) and target (y)
    X = df[['price_lag1', 'price_lag2', 'price_lag3']]
    y = df['price']
    
    return X, y

def train_model(X, y):
    """
    Train a simple linear regression model
    Input: X (features) and y (target)
    Returns: Trained model
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R² score: {train_score:.2f}")
    print(f"Testing R² score: {test_score:.2f}")
    
    return model

def make_predictions(model, last_prices):
    """
    Make predictions for the next 7 days
    Input: trained model and last known prices
    Returns: DataFrame with predictions
    """
    # Create future dates
    last_date = datetime.now()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    
    # Prepare features for prediction
    X_pred = pd.DataFrame({
        'price_lag1': [last_prices[0]] * 7,
        'price_lag2': [last_prices[1]] * 7,
        'price_lag3': [last_prices[2]] * 7
    })
    
    # Make predictions
    predictions = model.predict(X_pred)
    
    # Create result DataFrame
    results = pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions
    })
    
    return results

def plot_predictions(historical_data, predictions):
    """
    Plot historical prices and predictions
    Input: historical data and predictions DataFrames
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(historical_data['date'], historical_data['price'], 
             label='Historical Prices', color='blue')
    
    # Plot predictions
    plt.plot(predictions['date'], predictions['predicted_price'], 
             label='Predictions', color='red', linestyle='--')
    
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig('bitcoin_prediction.png')
    plt.close()

def main():
    """
    Main function to run the Bitcoin price prediction
    """
    print("Starting Bitcoin Price Prediction...")
    
    # Step 1: Load data
    print("\n1. Loading Bitcoin data...")
    df = load_bitcoin_data()
    if df is None:
        return
    
    # Step 2: Prepare data
    print("\n2. Preparing data...")
    X, y = prepare_data(df)
    
    # Step 3: Train model
    print("\n3. Training model...")
    model = train_model(X, y)
    
    # Step 4: Make predictions
    print("\n4. Making predictions...")
    last_prices = df['price'].tail(3).values
    predictions = make_predictions(model, last_prices)
    
    # Step 5: Plot results
    print("\n5. Plotting results...")
    plot_predictions(df, predictions)
    
    print("\nPrediction complete! Check bitcoin_prediction.png for results.")

if __name__ == '__main__':
    main()
