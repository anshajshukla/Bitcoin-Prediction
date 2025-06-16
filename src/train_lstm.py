import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from src.preprocess import load_data, get_train_test_split
from src.evaluate import calculate_metrics, print_metrics
from src.utils import plot_predictions, save_model, load_model
import os

# Function to create sequences for LSTM
def create_sequences(data, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm(train_data, sequence_length=60):
    """Trains an LSTM model."""
    print("\nTraining LSTM model...")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_data['close'].values.reshape(-1, 1))

    # Create sequences
    X_train, y_train = create_sequences(scaled_data, sequence_length)

    # Build the LSTM model
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10)

    return model, scaler

if __name__ == '__main__':
    # Load and split data
    df = load_data('data/btc.csv')
    train_df, test_df = get_train_test_split(df)

    sequence_length = 60 # Use last 60 hours to predict the next

    # Train the model
    lstm_model, scaler = train_lstm(train_df, sequence_length)

    # Prepare test data for prediction
    total_dataset = pd.concat((train_df['close'], test_df['close']), axis=0)
    inputs = total_dataset[len(total_dataset) - len(test_df) - sequence_length:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test, _ = create_sequences(inputs, sequence_length)

    # Make predictions
    predictions = lstm_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    # Create a predictions DataFrame
    pred_df = pd.DataFrame(predictions, index=test_df.index, columns=['predicted_close'])

    # Evaluate the model
    metrics = calculate_metrics(test_df['close'], pred_df['predicted_close'])
    print_metrics(metrics, "LSTM")

    # Plot the results
    plot_predictions(test_df['close'], pred_df['predicted_close'], "LSTM")

    # Save the model
    lstm_model.save(os.path.join('models', 'lstm.h5'))
    print("Model saved to models/lstm.h5")
