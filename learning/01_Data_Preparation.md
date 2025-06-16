# Data Preparation Guide

## Understanding Bitcoin Price Data

### Data Structure
The Bitcoin price data is stored in a CSV file (`data/btc.csv`) with the following structure:
```csv
date,price
2023-01-01,16500.00
2023-01-02,16600.00
...
```

### Data Loading Process
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
1. Reads the CSV file using pandas
2. Converts the date column to datetime format
3. Handles potential file not found errors

### Feature Engineering
We create three lag features to help predict future prices:
```python
df['price_lag1'] = df['price'].shift(1)  # Previous day's price
df['price_lag2'] = df['price'].shift(2)  # Price from 2 days ago
df['price_lag3'] = df['price'].shift(3)  # Price from 3 days ago
```

These features help the model understand price trends and patterns.

### Data Cleaning
1. Handle missing values:
```python
df = df.dropna()  # Remove rows with missing values
```

2. Data validation:
- Check for negative prices
- Verify date sequence
- Ensure no duplicate dates

### Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```
- 80% of data for training
- 20% of data for testing

## Best Practices
1. Always validate data before processing
2. Handle missing values appropriately
3. Normalize or scale data if needed
4. Keep track of data transformations
5. Document data sources and updates

## Common Issues and Solutions
1. Missing Data:
   - Use forward fill for small gaps
   - Use interpolation for larger gaps
   - Consider removing rows with too many missing values

2. Outliers:
   - Identify using statistical methods
   - Handle based on domain knowledge
   - Consider impact on predictions

3. Data Quality:
   - Check for data consistency
   - Verify price ranges
   - Ensure chronological order

## Data Visualization
Use these plots to understand your data:
1. Time series plot of prices
2. Price distribution
3. Autocorrelation plot
4. Rolling statistics

## Next Steps
1. Implement more advanced features
2. Add technical indicators
3. Consider external data sources
4. Implement data validation pipeline 