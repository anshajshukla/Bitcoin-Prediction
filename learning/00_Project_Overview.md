# 🧠 00: Project Overview

This project aims to build a robust Bitcoin price prediction system by leveraging an ensemble of time-series models. The core idea is that by combining the strengths of different models, we can create a more accurate and reliable forecast.

### Key Models Used:

1.  **SARIMA:** A classical statistical model that is excellent for capturing seasonality and trends in time-series data.
2.  **FBProphet:** Developed by Facebook, this model is designed to handle time-series data with strong seasonal effects and holiday impacts. It is robust to missing data and shifts in trends.
3.  **LSTM (Long Short-Term Memory):** A type of recurrent neural network (RNN) well-suited for sequence prediction problems. LSTMs can learn long-term dependencies, making them powerful for financial time-series.
4.  **XGBoost:** A gradient boosting algorithm that can be used for regression. We will frame the time-series problem as a supervised learning problem to use XGBoost.

### The Goal

The ultimate goal is to create an **ensemble model** that aggregates the predictions from these four individual models. This should, in theory, reduce the overall prediction error and provide a more stable forecast.

### The Pipeline

1.  **Data Preprocessing:** Load, clean, and feature-engineer the Bitcoin price data.
2.  **Model Training:** Train each of the four models on the historical data.
3.  **Evaluation:** Evaluate each model's performance using standard metrics (RMSE, MAE, MAPE).
4.  **Ensemble Prediction:** Combine the model predictions to generate a final forecast.
5.  **Visualization:** Plot the results to compare model performance visually.