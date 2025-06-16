# 🤝 06: Ensemble Model

An **Ensemble Model** in machine learning is a model that combines the predictions from two or more individual models to produce a final forecast. The primary goal of ensembling is to improve prediction accuracy and robustness.

### Why Use an Ensemble?

Different models have different strengths and weaknesses. For example:

-   **SARIMA** is good at capturing linear trends and seasonality.
-   **Prophet** is robust to changepoints and holidays.
-   **LSTM** can learn complex, long-term non-linear patterns.
-   **XGBoost** is excellent at finding non-linear relationships in structured feature data.

By combining their predictions, we can average out their individual errors and biases, often resulting in a forecast that is better than any single model on its own.

### Ensemble Strategy: Simple Averaging

For this project, we will use the simplest and often most effective ensemble strategy: **simple averaging**.

1.  **Generate Predictions:** We will take the test set and generate a forecast from each of our four trained models (SARIMA, Prophet, LSTM, and XGBoost).
2.  **Average the Results:** For each timestamp in the test set, we will simply calculate the average of the four predicted prices.

This averaged value will be our final ensemble prediction.

### Implementation in `ensemble.py`

The script will:

1.  **Load All Trained Models:** It will load `sarima.pkl`, `prophet.pkl`, `lstm.h5`, and `xgboost.json` from the `models/` directory.
2.  **Generate Individual Forecasts:** It will run the prediction logic for each model on the test data.
3.  **Combine Forecasts:** It will create a new DataFrame containing the predictions from all models and calculate the average.
4.  **Evaluate and Plot:** The performance of the ensemble forecast will be calculated and plotted against the actual values, allowing us to see if the ensemble approach provided an improvement.