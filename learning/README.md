# Bitcoin Price Prediction Project Documentation

## Project Overview
This project implements a simple Bitcoin price prediction system using machine learning. It uses historical Bitcoin price data to predict future prices using a linear regression model.

## Project Structure
```
bitcoin-timeseries-predictor/
├── data/
│   └── btc.csv                 # Historical Bitcoin price data
├── src/
│   ├── app.py                  # Flask web application
│   ├── templates/
│   │   └── index.html         # Web interface template
│   └── static/                # Static files (plots, CSS, etc.)
├── learning/
│   └── README.md              # This documentation
└── requirements.txt           # Project dependencies
```

## Implementation Details

### 1. Data Loading and Preparation
The project uses a simple CSV file containing Bitcoin price data with two columns:
- `date`: The date of the price
- `price`: The Bitcoin price in USD

The data preparation process includes:
- Loading the CSV file
- Converting dates to datetime format
- Creating lag features (previous days' prices)
- Handling missing values

### 2. Model Implementation
The project uses a simple Linear Regression model with the following features:
- Previous day's price (lag1)
- Price from 2 days ago (lag2)
- Price from 3 days ago (lag3)

The model is trained using scikit-learn's LinearRegression class, which:
- Splits data into training and testing sets (80/20 split)
- Fits the model to the training data
- Evaluates performance using R² score

### 3. Web Interface
The project includes a web interface built with Flask that provides:
- A visual chart of historical prices and predictions
- A table of historical prices (last 30 days)
- A table of price predictions (next 7 days)

The interface is built using:
- Flask for the backend
- Bootstrap for styling
- HTML templates for the frontend

### 4. API Endpoints
The project provides two main endpoints:
- `/`: The main page showing the prediction interface
- `/api/predictions`: JSON endpoint for getting prediction data

## How to Run the Project

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
- Place your Bitcoin price data in `data/btc.csv`
- Ensure the CSV has 'date' and 'price' columns

3. Run the Flask application:
```bash
cd src
python app.py
```

4. Access the web interface:
- Open your browser and go to `http://localhost:5000`

## Code Explanation

### Data Loading (app.py)
```python
def load_bitcoin_data():
    try:
        df = pd.read_csv('../data/btc.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        print("Error: Bitcoin data file not found!")
        return None
```
This function:
- Reads the CSV file
- Converts the date column to datetime format
- Handles file not found errors

### Data Preparation (app.py)
```python
def prepare_data(df):
    df['price_lag1'] = df['price'].shift(1)
    df['price_lag2'] = df['price'].shift(2)
    df['price_lag3'] = df['price'].shift(3)
    df = df.dropna()
    X = df[['price_lag1', 'price_lag2', 'price_lag3']]
    y = df['price']
    return X, y
```
This function:
- Creates lag features for previous prices
- Removes rows with missing values
- Prepares features (X) and target (y) for training

### Model Training (app.py)
```python
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
```
This function:
- Splits data into training and testing sets
- Creates and trains a linear regression model
- Returns the trained model

### Making Predictions (app.py)
```python
def make_predictions(model, last_prices):
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    X_pred = pd.DataFrame({
        'price_lag1': [last_prices[0]] * 7,
        'price_lag2': [last_prices[1]] * 7,
        'price_lag3': [last_prices[2]] * 7
    })
    predictions = model.predict(X_pred)
    return pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions
    })
```
This function:
- Generates future dates for predictions
- Creates prediction features using last known prices
- Makes predictions for the next 7 days
- Returns predictions in a DataFrame

### Web Interface (index.html)
The web interface is built using:
- Bootstrap for responsive design
- Tables for displaying data
- Charts for visualizing predictions
- Clean, modern styling

## Future Improvements
1. Add more advanced models (LSTM, Prophet)
2. Implement real-time data updates
3. Add more technical indicators
4. Improve prediction accuracy
5. Add user authentication
6. Implement API rate limiting
7. Add more visualization options 