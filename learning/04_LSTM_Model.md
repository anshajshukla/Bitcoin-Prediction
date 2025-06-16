# 🧠 04: LSTM Model

**LSTM** stands for **Long Short-Term Memory**. It is a special kind of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies in sequence data. This makes it extremely well-suited for time-series forecasting.

### How LSTMs Work

Traditional RNNs suffer from the "vanishing gradient" problem, which makes it difficult for them to remember information over long sequences. LSTMs solve this with a more complex internal structure called a **cell**.

Each LSTM cell has three "gates":

1.  **Forget Gate:** Decides what information to throw away from the cell state.
2.  **Input Gate:** Decides which new information to store in the cell state.
3.  **Output Gate:** Decides what to output based on the cell state.

This gating mechanism allows the network to selectively remember or forget information, enabling it to capture patterns over long periods.

### Why Use LSTMs?

-   **Sequence Learning:** They are specifically designed for sequential data like time series.
-   **Long-Term Dependencies:** They can capture complex, long-term patterns that simpler models might miss.
-   **Feature-Rich Data:** They can effectively use multiple input features (like the technical indicators we created) to make predictions.

### Implementation in `train_lstm.py`

1.  **Data Scaling:** Neural networks perform best when the input data is scaled. We will use `MinMaxScaler` from scikit-learn to scale our data to a range between 0 and 1.
2.  **Create Sequences:** We will transform our data into sequences. For example, we might use the last 60 hours of data (a sequence of length 60) to predict the price for the next hour.
3.  **Build the Model:** We will use Keras/TensorFlow to build the LSTM model architecture. This will include multiple LSTM layers, Dropout layers to prevent overfitting, and a Dense output layer to produce the final prediction.
4.  **Train the Model:** The model is compiled with an optimizer (like 'adam') and a loss function (like 'mean_squared_error') and then trained on the sequence data.
5.  **Forecast:** Making predictions involves taking the most recent sequence of data, predicting the next step, and then repeating this process for the entire test period.
6.  **Inverse Transform:** The predictions are scaled, so we must apply an inverse transformation to get them back into the original price scale.
7.  **Evaluate, Save, and Plot:** Finally, we evaluate the model, save it as an `.h5` file, and plot the results.