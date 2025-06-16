import pandas as pd
from sklearn.model_selection import train_test_split
from finta import TA

def load_data(filepath):
    """Loads and preprocesses the Bitcoin price data."""
    df = pd.read_csv(filepath)

    # Convert timestamp to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Sort by timestamp
    df = df.sort_index()

    # Feature Engineering with FinTA
    df['RSI'] = TA.RSI(df)
    df['MACD'] = TA.MACD(df)['MACD']
    df['SMA'] = TA.SMA(df, period=20)

    # Handle missing values created by indicators
    df = df.dropna()

    return df

def get_train_test_split(df, test_size=0.2):
    """Splits the data into training and testing sets."""
    train, test = train_test_split(df, test_size=test_size, shuffle=False)
    return train, test

if __name__ == '__main__':
    # Example usage
    data_path = 'data/btc.csv'
    df = load_data(data_path)
    train_df, test_df = get_train_test_split(df)

    print("Data loaded and preprocessed successfully.")
    print(f"Training data shape: {train_df.shape}")
    print(f"Testing data shape: {test_df.shape}")
    print("\nFirst 5 rows of training data:")
    print(train_df.head())
