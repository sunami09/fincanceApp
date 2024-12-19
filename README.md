### Introduction

Hi everyone! In this presentation, I’ll walk you through my project on stock price prediction using a Long Short-Term Memory, or LSTM, neural network. The goal of this project is to forecast stock prices based on historical data using deep learning techniques. Predicting stock prices is a notoriously challenging task due to the inherent volatility and randomness of the stock market. However, deep learning, and specifically LSTMs, offers a powerful approach to uncover patterns and trends in historical data, making this task more feasible. Let’s get started!

---

### ML Model Description

The primary goal of this project is to build a machine learning model that accurately predicts future stock prices using past data. The model aims to address key challenges in financial forecasting, such as market volatility and sudden price movements, by leveraging a data-driven approach. The problem we are solving here is to provide a predictive framework for stock prices that can aid investors and traders in making more informed decisions. For instance, it could help in planning strategies, mitigating risks, or understanding short-term market trends.

The model I used is an LSTM neural network. LSTMs are particularly well-suited for time-series data, such as stock prices, because they can learn long-term dependencies and patterns. They excel at capturing sequential relationships by using memory cells and gates to retain relevant information over time. This makes them more effective than traditional models like linear regression or ARIMA, which assume simpler or linear relationships in the data.

For this project, I used historical stock price data, specifically the closing prices of Apple Inc., retrieved from January 1, 2020, to December 31, 2023, using the Yahoo Finance API. The training style follows a supervised learning approach. The input consists of a sliding window of the last 60 days’ closing prices, and the target is the closing price of the following day. I also set aside 10 percent of the training data for validation, ensuring the model generalizes well to unseen data.

---

### Mathematical Methods and Algorithms

To preprocess the data, I normalized the stock prices using a MinMaxScaler to scale the values between zero and one. This step improves the model's performance by standardizing the data and preventing large gradients during training. For the loss function, I used Mean Squared Error, or MSE, which penalizes larger prediction errors more heavily. Gradient descent, combined with the Adam optimizer, updates the model weights during backpropagation to minimize this loss.

The LSTM model architecture itself uses memory cells with forget, input, and output gates to capture long-term dependencies. It also includes dropout layers to prevent overfitting and a dense output layer to predict a single value – the next day’s stock price. These components work together to provide a robust framework for sequential data.

In summary, the combination of normalization, MSE as the loss function, backpropagation, and LSTM’s unique architecture makes this model particularly well-suited for the complex task of stock price prediction. Together, these techniques help the model not only capture trends but also generalize well to unseen data, making it reliable for real-world applications.










### Step 1: Importing Libraries

"First, let’s import the necessary libraries. These libraries enable numerical computation, data preprocessing, model building, and visualization. [Run the cell.]

Here’s what each library does:

- **`numpy`**: Handles numerical computations, such as creating arrays and performing mathematical operations efficiently. This is crucial for preparing data for the machine learning model.
  
- **`pandas`**: Organizes and manipulates data in a tabular format. We use it to handle stock price data, making it easy to filter, process, and visualize.
  
- **`yfinance`**: Fetches historical stock price data directly from Yahoo Finance. This library simplifies the process of acquiring accurate and up-to-date market data.
  
- **`scikit-learn`**: Contains tools for preprocessing, including the `MinMaxScaler`, which normalizes data into a uniform range. This normalization is vital for the LSTM model to converge effectively.
  
- **`tensorflow.keras`**: The deep learning framework we use to build, train, and evaluate the LSTM model. It provides all the tools for defining the model architecture and optimizing it.
  
- **`matplotlib`**: Creates visualizations such as line plots, regression plots, and error curves to help interpret the model’s performance.

This combination of tools forms the backbone of our analysis, providing everything needed to process data, build a robust predictive model, and visualize the results effectively. By leveraging these libraries together, we streamline the entire workflow from data acquisition to model evaluation."

















### Step 2: Retrieving Data

"Next, we fetch historical stock prices using the `get_stock_data` function. [Run the cell.]

This function:

- **Downloads stock data**: Retrieves data for a specified stock ticker symbol (default: `AAPL` for Apple) and date range.
- **Returns a DataFrame**: The output is a Pandas DataFrame with key columns such as `Close`, `Volume`, `High`, `Low`, and `Open`, which represent daily trading activity.
- **Focuses on `Close` Prices**: While the DataFrame contains multiple fields, we’ll focus on the `Close` column, as it represents the stock’s closing price for each trading day—an ideal target for predictions.

Here’s the last 5 rows of the data, showing recent trading activity. This ensures the data is accurate and up-to-date for modeling purposes."


















### Step 3: Preparing Data

"Now, we prepare the data using the `prepare_data` function. [Run the cell.]

This function performs three key tasks:

1. **Normalization**: It scales the closing prices to a range between 0 and 1 using `MinMaxScaler`, which is critical for improving the LSTM model's performance.
2. **Creating Sliding Windows**: It generates input-output pairs where the input is a sequence of `look_back` days (e.g., 60 days), and the output is the target price for the next day. This is how the LSTM learns patterns in the data.
3. **Reshaping for LSTM**: The input data is reshaped into a 3D array with dimensions `[samples, time steps, features]` to fit the requirements of the LSTM architecture.

The outputs of this function are:
- `X_train`: Input sequences (3D array).
- `y_train`: Target values corresponding to the sequences.
- `scaler`: A fitted `MinMaxScaler` instance to reverse the scaling later for interpretation of results.

This step transforms raw historical data into a structured format that the LSTM model can learn from effectively."

---

### Explanation of the Code:

```python
def prepare_data(df, look_back=60, scaler=None):
    """Prepare data for LSTM model. If scaler is provided, use it; else fit a new one."""
```

**Overview**: 
This function processes the `Close` column from the stock price DataFrame to create normalized input-output pairs suitable for training an LSTM model.

---

1. **Extracting the Closing Prices**:
   ```python
   data = df['Close'].values.reshape(-1, 1)
   ```
   - Extracts the `Close` prices from the DataFrame and reshapes them into a 2D array.
   - Necessary for applying the scaler and preparing input sequences.

2. **Normalization**:
   ```python
   if scaler is None:
       scaler = MinMaxScaler()
       data_scaled = scaler.fit_transform(data)
   else:
       data_scaled = scaler.transform(data)
   ```
   - If no scaler is provided, a new `MinMaxScaler` is instantiated and fit to the data, scaling values to a range of [0, 1].
   - If a pre-fitted scaler is provided, it uses that to transform the data. This ensures consistency during training and testing.

3. **Creating Input and Output Pairs**:
   ```python
   X, y = [], []
   for i in range(look_back, len(data_scaled)):
       X.append(data_scaled[i-look_back:i, 0])
       y.append(data_scaled[i, 0])
   X, y = np.array(X), np.array(y)
   ```
   - Uses a sliding window approach to create sequences:
     - `X`: Contains `look_back` days of scaled closing prices as input.
     - `y`: Contains the next day's scaled closing price as the target.
   - Example:
     - If `look_back=3` and `data_scaled = [0.1, 0.2, 0.3, 0.4]`, the pairs are:
       - `X = [[0.1, 0.2, 0.3]]`
       - `y = [0.4]`

4. **Reshaping for LSTM**:
   ```python
   X = np.reshape(X, (X.shape[0], X.shape[1], 1))
   ```
   - The LSTM model requires input in the shape `[samples, time steps, features]`.
   - Here:
     - `samples`: Number of sequences.
     - `time steps`: `look_back` days.
     - `features`: Single feature (closing price).

5. **Return Values**:
   ```python
   return X, y, scaler
   ```
   - `X`: Input sequences (3D array).
   - `y`: Target values (1D array).
   - `scaler`: Fitted scaler for scaling and inverse transformations.

---

### Example Output:

**If `look_back=3` and `data = [100, 101, 102, 103, 104]`:**
- Scaled `data_scaled = [0.0, 0.25, 0.5, 0.75, 1.0]`.
- Generated pairs:
  - `X = [[[0.0], [0.25], [0.5]], [[0.25], [0.5], [0.75]]]`
  - `y = [0.75, 1.0]`.

This step ensures that the data is structured and normalized, ready for effective learning by the LSTM model.





















### Step 4: Building the Model

"Let’s build our LSTM model using the `create_model` function. [Run the cell.]

This model is designed to capture temporal dependencies in the stock price data. It consists of:

1. **Two LSTM layers**: These layers learn patterns and relationships over time from the sequences of stock prices.
2. **Dropout layers**: These help reduce overfitting by randomly disabling 20% of the neurons during training.
3. **A dense output layer**: This produces a single prediction (the next day’s stock price).
4. **Compilation settings**:
   - **Optimizer**: Adam, an adaptive optimizer that adjusts learning rates dynamically.
   - **Loss function**: Mean Squared Error (MSE), which penalizes larger errors more heavily.
   - **Metrics**: Mean Absolute Error (MAE), which tracks the average magnitude of errors during training.

This architecture is well-suited for time-series data like stock prices, where learning from sequential patterns is crucial."

---

### Code Explanation:

```python
def create_model(look_back):
    """Create LSTM model."""
```
- **Function Definition**:
  - Takes one parameter:
    - `look_back`: The number of past days to consider for each input sequence.
  - Returns a compiled LSTM model.

---

1. **Defining the Model**:
   ```python
   model = Sequential([
   ```
   - Creates a sequential model, where layers are added in a linear stack.

---

2. **First LSTM Layer**:
   ```python
   LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
   ```
   - **LSTM**: A specialized type of recurrent neural network (RNN) that learns temporal dependencies.
   - **Parameters**:
     - `units=50`: Specifies 50 neurons (units) in the layer, which determine the dimensionality of the output.
     - `return_sequences=True`: Ensures the layer outputs a sequence rather than a single value, enabling the second LSTM layer to process it.
     - `input_shape=(look_back, 1)`: Defines the input shape for the first layer:
       - `look_back`: Number of past days in each sequence.
       - `1`: Single feature (closing price).

---

3. **First Dropout Layer**:
   ```python
   Dropout(0.2),
   ```
   - **Purpose**: Reduces overfitting by randomly deactivating 20% of the neurons during training.
   - Helps improve the model's ability to generalize to unseen data.

---

4. **Second LSTM Layer**:
   ```python
   LSTM(units=50),
   ```
   - Another LSTM layer with 50 neurons.
   - **Note**: `return_sequences=False` (default), as this layer is not followed by another LSTM layer. It outputs a single vector summarizing the sequence.

---

5. **Second Dropout Layer**:
   ```python
   Dropout(0.2),
   ```
   - Again, reduces overfitting by randomly deactivating 20% of the neurons.

---

6. **Dense Output Layer**:
   ```python
   Dense(units=1)
   ```
   - A fully connected (dense) layer with one neuron.
   - Outputs a single value, which represents the predicted stock price.

---

7. **Model Compilation**:
   ```python
   model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   ```
   - **Optimizer**: Adam, an efficient adaptive learning rate optimization algorithm.
   - **Loss Function**: MSE, which minimizes the squared differences between predicted and actual values.
   - **Metrics**: MAE, which tracks the average magnitude of prediction errors.

---

8. **Return Statement**:
   ```python
   return model
   ```
   - Returns the compiled LSTM model, ready for training.

---

### Summary of the Model Architecture:

| **Layer**          | **Type**   | **Output Shape**            | **Parameters**          |
|---------------------|------------|-----------------------------|--------------------------|
| Input              | LSTM       | `(batch_size, look_back, 50)` | 10,400 (from weights and biases) |
| Dropout            | Dropout    | `(batch_size, look_back, 50)` | 0                        |
| Second LSTM        | LSTM       | `(batch_size, 50)`          | 20,200                   |
| Dropout            | Dropout    | `(batch_size, 50)`          | 0                        |
| Output             | Dense      | `(batch_size, 1)`           | 51 (weights + bias)      |

---

### Why This Model Works for Stock Prediction:

1. **LSTMs**:
   - Designed to handle sequential data.
   - Use memory cells to retain information about previous time steps, making them ideal for learning temporal patterns in stock prices.

2. **Dropout**:
   - Reduces overfitting, ensuring the model performs well on unseen data.

3. **Dense Output**:
   - Outputs a single stock price prediction, simplifying the results.

This architecture strikes a balance between capturing complex patterns in stock prices and maintaining generalization to unseen data.

















Step 5: Training the Model
"Next, we proceed to train the model. [Run the cell.]

The training process involves running the model for 50 epochs with a batch size of 32. During each epoch, 10% of the data is set aside for validation, allowing us to monitor the model's performance on unseen data throughout the training phase. As the training progresses, the loss (measured using Mean Squared Error) and the Mean Absolute Error (MAE) steadily decrease. This indicates that the model is effectively learning patterns from the data and improving its predictions. At the end of training, the RMSE on the training data is calculated as 4.77, signifying strong predictive accuracy on the historical stock prices."

Step 6: Visualizing Training Performance
"Let’s now visualize the training and validation performance. [Run the cell.]

The generated plots display the loss and MAE for both the training and validation sets over the epochs. The loss plot shows a consistent downward trend, reflecting that the model is minimizing errors effectively as training progresses. Similarly, the MAE plot reveals decreasing prediction errors, highlighting improvements in the model’s accuracy. The small gap between the training and validation metrics suggests minimal overfitting, confirming that the model is generalizing well to unseen data. These trends reinforce the reliability of the model in capturing meaningful patterns from the stock price data."




















Step 7: Evaluating Test Data
"Now, we evaluate the model's performance on test data from 2024. [Run the cell.]

The evaluation is conducted using key metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and MAPE (Mean Absolute Percentage Error). The RMSE of 8.48 indicates the average deviation of predicted stock prices from the actual values. The MAE, which measures the average magnitude of errors without considering their direction, is 7.02, reflecting a small overall prediction error. Finally, the MAPE of 3.32% demonstrates the accuracy of predictions relative to the actual stock prices, confirming that the model performs well on unseen data with minimal errors. These results validate the model's ability to generalize effectively to new data while maintaining high predictive accuracy."

Step 8: Plotting Test Results
"Here, we plot the actual and predicted stock prices for 2024 to visually assess the model's performance. [Run the cell.]

The plot shows a clear alignment between the actual and predicted prices, demonstrating the model’s ability to capture underlying trends in stock movements. The predicted prices closely follow the trajectory of the actual prices, accurately reflecting patterns and changes. While there are minor deviations during periods of high volatility, the model’s overall performance remains robust. This visual representation reinforces the quantitative metrics, illustrating the reliability of the predictions and the model's effectiveness in learning from historical data to predict future prices."

Step 9: Regression Plot
"This regression plot provides a comparison between the predicted and actual prices. [Run the cell.]

The plot shows individual predictions as points, with a red dashed line representing the ideal scenario where predicted prices perfectly match the actual values. Most points cluster closely around this line, indicating strong predictive accuracy across the test data. The clustering reflects the model’s consistency in producing accurate predictions. However, some outliers are present, highlighting areas where the model struggled, such as during periods of unusual price volatility. These outliers suggest opportunities for further improvement, potentially by incorporating additional features or refining the model architecture. Overall, the plot confirms the model’s reliability and its ability to generalize well to unseen data."




















Step 10: Predicting Future Prices
"Finally, we use the model to predict stock prices for the next seven days based on the most recent historical data. [Run the cell.]

The prediction process involves taking the last 60 days of closing prices from the combined training and test data as input. These prices are normalized using the same scaler used during training to maintain consistency. The model iteratively predicts one day at a time, and each prediction is added to the sliding input window, replacing the oldest value to generate predictions for subsequent days. This process continues for seven iterations to produce forecasts for the entire week ahead.

The predicted prices are as follows: [235.01, 234.44, 232.97, 230.95, 228.59, 226.03, 223.37]. These values indicate a short-term downward trend in stock prices, reflecting a gradual decrease over the next week. The predictions align with the trends and patterns observed in the input data, highlighting the model's ability to capture and extend learned patterns into the future. However, it is essential to note that these predictions are based solely on historical price movements and do not account for external factors like news, market events, or macroeconomic conditions that could impact stock prices."




















