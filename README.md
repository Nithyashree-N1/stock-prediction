# stock-prediction
# This is a simplified Python code for stock price prediction using a Jupyter Notebook, focusing on Tata Steel Limited (TSL) as an example.  
import pandas as pd
import numpy as np
import yfinance as yf  # Yahoo Finance API

# Define the stock symbol and date range
symbol = "TSLA"
start_date = "2020-01-01"
end_date = "2021-12-31"

# Fetch historical stock price data from Yahoo Finance
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Feature engineering (example: using daily closing prices)
stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
stock_data['Volatility'] = stock_data['Daily_Return'].rolling(window=30).std()

# Prepare the data for modeling (X: features, y: target)
X = stock_data[['Open', 'High', 'Low', 'Volume', 'Volatility']].dropna()
y = X.pop('Close')

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train a simple linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model (example: using Mean Absolute Error)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# Visualize the actual vs. predicted prices (requires matplotlib)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index[train_size:], y_test, label='Actual Prices', color='blue')
plt.plot(stock_data.index[train_size:], predictions, label='Predicted Prices', color='red')
plt.title(f"{symbol} Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
