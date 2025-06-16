# Import necessary libraries
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive 'Agg'
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import requests
import time

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.dirname(__file__)

# Create Flask application
app = Flask(__name__, 
           static_folder=os.path.join(SRC_DIR, 'static'),
           template_folder=os.path.join(SRC_DIR, 'templates'))

# Ensure data and static directories exist
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
STATIC_DIR = os.path.join(SRC_DIR, 'static')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

def fetch_bitcoin_data():
    """Fetch Bitcoin price data from CoinGecko API"""
    try:
        # Get data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Convert to Unix timestamps
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # CoinGecko API endpoint
        url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': start_timestamp,
            'to': end_timestamp
        }
        
        # Make API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse response
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['date', 'price']]
        
        # Save to CSV for future use
        data_file = os.path.join(DATA_DIR, 'btc.csv')
        df.to_csv(data_file, index=False)
        
        print("Successfully fetched Bitcoin data")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        # Try to load from local file if API fails
        try:
            data_file = os.path.join(DATA_DIR, 'btc.csv')
            df = pd.read_csv(data_file)
            df['date'] = pd.to_datetime(df['date'])
            print("Loaded data from local file")
            return df
        except FileNotFoundError:
            print("No local data file found")
            return None

def prepare_data(df):
    """Prepare data for training"""
    # Create simple features using last 3 days
    df['price_lag1'] = df['price'].shift(1)
    df['price_lag2'] = df['price'].shift(2)
    df['price_lag3'] = df['price'].shift(3)
    
    # Remove missing values
    df = df.dropna()
    
    # Prepare features and target
    X = df[['price_lag1', 'price_lag2', 'price_lag3']]
    y = df['price']
    
    return X, y

def train_model(X, y):
    """Train a simple linear regression model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Training R² score: {train_score:.2f}")
    print(f"Testing R² score: {test_score:.2f}")
    
    return model

def make_predictions(model, last_prices):
    """Make predictions for next 7 days"""
    # Generate future dates
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

def create_plot(historical_data, predictions):
    """Create and save prediction plot"""
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
    
    # Save plot
    plot_filename = 'bitcoin_prediction.png'
    plot_path = os.path.join(STATIC_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

@app.route('/')
def home():
    """Main page with predictions"""
    # Fetch and prepare data
    df = fetch_bitcoin_data()
    if df is None:
        return "Error: Could not fetch Bitcoin data", 500
    
    X, y = prepare_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Make predictions
    last_prices = df['price'].tail(3).values
    predictions = make_predictions(model, last_prices)
    
    # Create plot
    plot_filename = create_plot(df, predictions)
    
    # Prepare data for template
    historical_data = df.tail(30).to_dict('records')  # Last 30 days
    prediction_data = predictions.to_dict('records')
    
    return render_template('index.html',
                         historical_data=historical_data,
                         prediction_data=prediction_data,
                         plot_path=plot_filename)

@app.route('/api/predictions')
def get_predictions():
    """API endpoint for predictions"""
    df = fetch_bitcoin_data()
    if df is None:
        return jsonify({"error": "Could not fetch Bitcoin data"}), 500
    
    X, y = prepare_data(df)
    model = train_model(X, y)
    last_prices = df['price'].tail(3).values
    predictions = make_predictions(model, last_prices)
    
    return jsonify(predictions.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True, port=5000) 