# Bitcoin Price Prediction Using Time-Series Ensemble Models

This project aims to build a Bitcoin price forecasting system that combines traditional and deep learning models to generate accurate predictions.

## 🎯 Goal

- Combine SARIMA, FBProphet, LSTM, and XGBoost for price forecasting.
- Process historical Bitcoin price data.
- Output predictions along with performance metrics (RMSE, MAE, MAPE).
- Create an ensemble prediction to improve accuracy.

## 📁 Project Structure

```
bitcoin-timeseries-predictor/
├── data/
│   └── btc.csv
├── models/
│   ├── sarima.pkl
│   ├── prophet.pkl
│   ├── lstm.h5
│   └── xgboost.json
├── src/
│   ├── preprocess.py
│   ├── train_sarima.py
│   ├── train_prophet.py
│   ├── train_lstm.py
│   ├── train_xgboost.py
│   ├── ensemble.py
│   ├── evaluate.py
│   └── utils.py
├── notebooks/
│   └── EDA.ipynb
├── learning/
│   └── ... (markdown explanations)
├── main.py
└── README.md
```

## 🚀 How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the pipeline:**
    ```bash
    python main.py
    ```
