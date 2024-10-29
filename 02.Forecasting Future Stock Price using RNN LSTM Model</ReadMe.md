# __Project Overview__
In this project, I developed a Recurrent Neural Network (RNN) model to forecast stock prices, specifically focusing on Tesla Inc. (TSLA). 
This model predicts the closing price based on historical stock data from the past decade. 
It highlights my skills in data analysis, preprocessing, and neural network training, aiming to showcase a realistic approach to time-series forecasting.


## __Modeling Approach__
I used a Long Short-Term Memory (LSTM) network to capture temporal patterns in stock price movements. The model was trained on Tesla's historical data from 2013 to 2023, using mean squared error as the loss function. 


## __Model Performance:__
The model's predictions (shown by the orange line) closely follow the validation data, indicating a fair fit. 
Performance was evaluated using Mean Absolute Percentage Error (MAPE), with an MAPE score of around 30%, demonstrating fair predictive accuracy on unseen data.

In general, here’s a loose guide to interpreting MAPE values:

MAPE < 10%: Highly accurate predictions 

10% ≤ MAPE < 20%: Good predictions, but with noticeable error

20% ≤ MAPE < 50%: Fair predictions, with significant error

MAPE ≥ 50%: Poor predictions, with very high deviation from actual values


Considering that we used in our modeling only the last/close value on the stock and got fair to good results showcases the power of these models.
For more accurate predictions, more factors could be taken in consideration, like open, high, and low prices, trading Volume etc.



![image](https://github.com/user-attachments/assets/6fd62206-524a-4dbd-8dff-e3c98c49c0d3)

- The orange line represents model predictions, while the validation data line shows the actual historical price.

## __How to Use:__
1. Download the notebook and the accompanying CSV file, which contains historical stock data.
2. To test this on other stocks, you can use the `yfinance` library to download historical data. For example:

   ```python
   import yfinance as yf

   # Download Apple stock data from the last 10 years
   data = yf.download("AAPL", start="2014-01-01", end="2024-01-01")
   print(data.head())









