# Model Implementation Guide

## Linear Regression Model

### Model Overview
We use a simple Linear Regression model to predict Bitcoin prices based on historical data. The model uses three features:
- Previous day's price (lag1)
- Price from 2 days ago (lag2)
- Price from 3 days ago (lag3)

### Implementation
```python
def train_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model
```

### Model Evaluation
1. R² Score:
   - Measures how well the model fits the data
   - Range: 0 to 1 (higher is better)
   - Calculated for both training and test sets

2. Mean Squared Error (MSE):
   - Measures average squared difference between predictions and actual values
   - Lower values indicate better performance

### Making Predictions
```python
def make_predictions(model, last_prices):
    # Generate future dates
    future_dates = [datetime.now() + timedelta(days=i) for i in range(1, 8)]
    
    # Prepare features
    X_pred = pd.DataFrame({
        'price_lag1': [last_prices[0]] * 7,
        'price_lag2': [last_prices[1]] * 7,
        'price_lag3': [last_prices[2]] * 7
    })
    
    # Make predictions
    predictions = model.predict(X_pred)
    
    return pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions
    })
```

## Model Limitations
1. Assumes linear relationship between features and target
2. Sensitive to outliers
3. May not capture complex market patterns
4. Limited to short-term predictions

## Improving the Model
1. Feature Engineering:
   - Add more technical indicators
   - Include market sentiment data
   - Consider external factors

2. Model Selection:
   - Try more complex models (LSTM, Prophet)
   - Implement ensemble methods
   - Use different time horizons

3. Hyperparameter Tuning:
   - Grid search for optimal parameters
   - Cross-validation
   - Regularization

## Best Practices
1. Always validate model assumptions
2. Monitor model performance over time
3. Keep track of model versions
4. Document model limitations
5. Regular retraining with new data

## Common Issues
1. Overfitting:
   - Use regularization
   - Increase training data
   - Simplify model

2. Underfitting:
   - Add more features
   - Try more complex models
   - Check data quality

3. Data Leakage:
   - Proper train-test split
   - Time-based validation
   - Feature selection

## Next Steps
1. Implement more advanced models
2. Add model evaluation metrics
3. Create model monitoring system
4. Implement automated retraining 