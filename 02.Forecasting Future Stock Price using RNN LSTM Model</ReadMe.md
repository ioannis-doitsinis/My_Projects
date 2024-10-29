# __Project Overview__
In this project, I developed a Recurrent Neural Network (RNN) model to forecast stock prices, specifically focusing on Tesla Inc. (TSLA). 
This model predicts the closing price based on historical stock data from the past decade. 
It highlights my skills in data analysis, preprocessing, and neural network training, aiming to showcase a realistic approach to time-series forecasting.


## __Modeling Approach__
I used a Long Short-Term Memory (LSTM) network to capture temporal patterns in stock price movements. The model was trained on Tesla's historical data from 2014 to 2024, using mean squared error as the loss function. 


## __Model Performance:__
The model's predictions (shown by the orange line) closely follow the validation data, indicating a good fit. Performance was evaluated using Root Mean Squared Error (RMSE), with an RMSE score of 1.13, demonstrating reliable predictive accuracy on unseen data.


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









