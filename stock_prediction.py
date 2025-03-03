import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Step 1: Fetch Stock Data from Yahoo Finance
stock_symbol = "AAPL"  # Change this to any stock ticker
start_date = "2015-01-01"
end_date = "2024-01-01"

data = yf.download(stock_symbol, start=start_date, end=end_date)
df = data[['Close']].dropna()

# Step 2: Visualize Stock Price Trends
plt.figure(figsize=(10,5))
plt.plot(df['Close'], label="Stock Price")
plt.title(f"{stock_symbol} Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.legend()
plt.show()

# Step 3: Check Stationarity using Dickey-Fuller Test
def check_stationarity(data):
    result = adfuller(data)
    print("Dickey-Fuller Test p-value:", result[1])
    if result[1] < 0.05:
        print("The data is stationary.")
    else:
        print("The data is NOT stationary. Differencing is needed.")

check_stationarity(df['Close'])

# Step 4: Apply Differencing if Needed
df['Close_diff'] = df['Close'].diff().dropna()
check_stationarity(df['Close_diff'].dropna())

# Step 5: Identify ARIMA Parameters (p, d, q)
plt.figure(figsize=(10,5))
plot_acf(df['Close_diff'].dropna(), lags=40)
plot_pacf(df['Close_diff'].dropna(), lags=40)
plt.show()

# Step 6: Train ARIMA Model
model = ARIMA(df['Close'], order=(2,1,1))  # Adjust (p,d,q) based on ACF/PACF analysis
model_fit = model.fit()
print(model_fit.summary())

# Step 7: Forecast Future Prices
forecast_days = 30
forecast = model_fit.forecast(steps=forecast_days)
future_dates = pd.date_range(df.index[-1], periods=forecast_days+1, freq="B")[1:]
forecast_df = pd.DataFrame(forecast, index=future_dates, columns=['Predicted Close'])

# Step 8: Plot Predictions
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label="Actual Price", color="blue")
plt.plot(forecast_df, label="Predicted Price", color="red", linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title(f"{stock_symbol} Stock Price Prediction (Next {forecast_days} Days)")
plt.legend()
plt.show()

# Step 9: Evaluate Model Performance
y_true = df['Close'][-forecast_days:]
y_pred = forecast[:forecast_days]

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
