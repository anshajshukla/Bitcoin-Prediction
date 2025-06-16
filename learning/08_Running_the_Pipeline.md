# 🚀 08: Running the Pipeline

Now that all the components are built, you can run the entire forecasting pipeline with a single command.

### The `main.py` Script

This script acts as the entry point to our project. It automates the following process:

1.  **Sequentially trains each model:**
    -   `train_sarima.py`
    -   `train_prophet.py`
    -   `train_lstm.py`
    -   `train_xgboost.py`

2.  **Saves each trained model** to the `models/` directory.

3.  **Runs the ensemble script** (`ensemble.py`), which loads the trained models, generates the final ensemble forecast, and evaluates all models.

### How to Run

1.  **Open your terminal or command prompt.**

2.  **Navigate to the project's root directory:**
    ```bash
    cd path/to/bitcoin-timeseries-predictor
    ```

3.  **Make sure you have installed all the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute the main script:**
    ```bash
    python main.py
    ```

### What to Expect

As the script runs, you will see output from each model's training process, including model summaries and performance metrics. At the end, you will see a final comparison of all models.

All the prediction plots will be saved in the `plots/` directory for you to review.