# 🔥 05: XGBoost Model

**XGBoost** (eXtreme Gradient Boosting) is a highly efficient and effective implementation of the gradient boosting algorithm. While not a traditional time-series model, it can be adapted for forecasting by transforming the time-series problem into a supervised learning problem.

### How XGBoost Works for Time-Series

1.  **Feature Creation:** We cannot use XGBoost on a raw time series directly. Instead, we must create features from it. Common time-series features include:
    *   **Lag Features:** Past values of the time series (e.g., the price 1 hour ago, 2 hours ago, etc.).
    *   **Time-based Features:** Features derived from the timestamp, such as the hour of the day, day of the week, or month of the year.
    *   **Rolling Window Features:** Statistics calculated over a rolling window, like a moving average or standard deviation.

2.  **Supervised Learning:** Once we have these features, the problem becomes: "Given the features at time `t`, what is the price at time `t+1`?" This is a standard regression problem that XGBoost can solve.

3.  **Gradient Boosting:** XGBoost works by building a series of decision trees, where each new tree corrects the errors of the previous one. This iterative process makes it a very powerful predictive model.

### Why Use XGBoost?

-   **Performance:** It is known for its high performance and is a popular choice in machine learning competitions.
-   **Handles Complex Relationships:** It can capture complex, non-linear relationships between features and the target variable.
-   **Feature Importance:** It can provide insights into which features are most important for making predictions.

### Implementation in `train_xgboost.py`

1.  **Feature Engineering:** The script will create lag features and time-based features from our training data.
2.  **Train the Model:** We will train an `XGBRegressor` model on the engineered features.
3.  **Forecast:** Predictions are made one step at a time. For each step in the test set, we create the necessary features and feed them to the model.
4.  **Evaluate, Save, and Plot:** The model's performance is evaluated, it's saved to a file, and the predictions are plotted.