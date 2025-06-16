# 📈 02: SARIMA Model

**SARIMA** stands for **Seasonal AutoRegressive Integrated Moving Average**. It is a powerful statistical model used for time-series forecasting that can capture complex patterns.

Let's break down the acronym:

-   **AR (AutoRegressive):** The model assumes that the current value is dependent on its own previous values. The `p` parameter defines how many past values to include.
-   **I (Integrated):** This component involves differencing the time-series data to make it stationary (i.e., its statistical properties like mean and variance are constant over time). The `d` parameter represents the number of times the data is differenced.
-   **MA (Moving Average):** The model assumes that the current value is dependent on the error terms of previous values. The `q` parameter defines the number of past error terms to include.
-   **S (Seasonal):** The 'S' in SARIMA adds a seasonal component. It's an extension of ARIMA that explicitly supports time-series data with a seasonal component. It adds another set of parameters `(P, D, Q, m)` for the seasonal part of the model.

### Why Use SARIMA?

-   **Seasonality:** Financial data like Bitcoin prices often exhibits seasonal patterns (e.g., weekly, monthly). SARIMA is specifically designed to handle this.
-   **Established & Interpretable:** It's a well-understood statistical method, and the model's parameters can be interpreted.

### Implementation in `train_sarima.py`

1.  **Load Data:** The script starts by loading the preprocessed data.
2.  **Find Optimal Parameters:** We will use the `pmdarima` library's `auto_arima` function to automatically find the best `(p, d, q)(P, D, Q, m)` parameters for our model. This saves us from a manual and time-consuming grid search.
3.  **Train the Model:** We fit the SARIMA model with the optimal parameters on the training data (`close` price).
4.  **Forecast:** The trained model is used to make predictions on the test set.
5.  **Evaluate:** We calculate RMSE, MAE, and MAPE to assess the model's accuracy.
6.  **Save and Plot:** The trained model is saved, and the predictions are plotted against the actual values.