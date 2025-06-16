# 📊 01: Data Preprocessing

Data preprocessing is a critical first step in any machine learning pipeline. For time-series forecasting, this involves several key tasks to transform raw data into a format suitable for our models.

### The `preprocess.py` Script

This script handles the following steps:

1.  **Load Data:** It loads the historical Bitcoin data from `data/btc.csv` using the `pandas` library.

2.  **Timestamp Conversion:** The `timestamp` column is converted from a string format into a `datetime` object. This is essential for time-series analysis, as it allows us to use time-based indexing and slicing.

3.  **Set Index:** The `timestamp` column is set as the DataFrame index. This makes it easier to work with the time-series data and is a requirement for many time-series libraries.

4.  **Feature Engineering with FinTA:** To enrich our dataset, we add several technical indicators using the `FinTA` library. These indicators are calculated from the OHLCV (Open, High, Low, Close, Volume) data and can provide valuable signals for our models. Examples include:
    *   RSI (Relative Strength Index)
    *   MACD (Moving Average Convergence Divergence)
    *   SMA (Simple Moving Average)

5.  **Handle Missing Values:** The script checks for and handles any missing values. For time-series data, a common strategy is to forward-fill (`ffill`) or back-fill (`bfill`) missing data points to maintain the sequence.

6.  **Train-Test Split:** The data is split into a training set (80%) and a testing set (20%). We use the older data for training and the most recent data for testing to simulate a real-world forecasting scenario.