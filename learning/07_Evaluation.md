# 📊 07: Model Evaluation

Evaluating the performance of our forecasting models is crucial to understanding their accuracy and reliability. We use several standard regression metrics to measure how close our predictions are to the actual values.

### Key Metrics Used

Our `evaluate.py` script calculates the following metrics:

1.  **RMSE (Root Mean Squared Error):**
    -   **What it is:** The square root of the average of the squared differences between the predicted and actual values.
    -   **Why it's useful:** It penalizes large errors more than small ones, so it's a good measure of how well the model predicts large price swings.
    -   **Goal:** Lower is better.

2.  **MAE (Mean Absolute Error):**
    -   **What it is:** The average of the absolute differences between the predicted and actual values.
    -   **Why it's useful:** It gives a clear, interpretable measure of the average error in the same units as the data (in our case, USD).
    -   **Goal:** Lower is better.

3.  **MAPE (Mean Absolute Percentage Error):**
    -   **What it is:** The average of the absolute percentage differences between the predicted and actual values.
    -   **Why it's useful:** It expresses the error as a percentage, which can be easier to understand and compare across different datasets or models.
    -   **Goal:** Lower is better.

### Comparing Models

After running the entire pipeline, the `ensemble.py` script will print a summary table of these metrics for all four individual models and the final ensemble model. This allows us to directly compare their performance and determine if our ensemble strategy was successful in improving the forecast accuracy.