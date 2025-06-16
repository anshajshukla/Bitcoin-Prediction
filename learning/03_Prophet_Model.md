# 📅 03: Prophet Model

**Prophet** is an open-source forecasting tool developed by Facebook. It is designed to be easy to use and to produce high-quality forecasts for time-series data that has strong seasonal effects and several seasons of historical data.

### How Prophet Works

Prophet decomposes the time series into three main components:

1.  **Trend:** It models non-periodic changes in the value of the time series.
2.  **Seasonality:** It models periodic changes (e.g., weekly, yearly).
3.  **Holidays:** It models the effects of holidays or special events that occur on irregular schedules.

Prophet fits these components to the data and then uses them to make predictions.

### Why Use Prophet?

-   **Automatic Seasonality:** Prophet automatically detects weekly and yearly seasonalities and can also model daily seasonality if the data is at that resolution.
-   **Robust to Missing Data:** It can handle missing data and outliers gracefully.
-   **Fast and Easy to Use:** It requires minimal data preprocessing and tuning compared to models like SARIMA.
-   **Changepoint Detection:** It can automatically detect trend changepoints in the data.

### Implementation in `train_prophet.py`

1.  **Load Data:** The script loads the preprocessed data.
2.  **Format for Prophet:** Prophet requires the data to be in a specific format: a DataFrame with two columns named `ds` (datestamp) and `y` (the value to be forecasted). We will transform our data accordingly.
3.  **Train the Model:** We instantiate and fit the Prophet model on the training data.
4.  **Forecast:** We create a `future` DataFrame that includes the timestamps from our test set and use the model to predict values for this period.
5.  **Evaluate:** We calculate RMSE, MAE, and MAPE for the predictions.
6.  **Save and Plot:** The model is saved, and the forecast is plotted, including the trend and seasonality components.