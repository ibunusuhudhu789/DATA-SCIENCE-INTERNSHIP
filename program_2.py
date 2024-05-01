import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Step 1: Data Collection
stock_data = pd.read_csv('hackathon.csv')

# Step 2: Data Preprocessing
stock_data['ID'] = pd.to_datetime(stock_data['ID'])
stock_data.set_index('ID', inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
plt.plot(stock_data['STORECODE'], color='blue')
plt.title('Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Step 4: Time Series Decomposition
decomposition = seasonal_decompose(stock_data['Close'], model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(10, 8))
plt.subplot(411)
plt.plot(stock_data['Close'], label='Original', color='blue')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label='Trend', color='red')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality', color='green')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label='Residuals', color='yellow')
plt.legend(loc='upper left')
plt.tight_layout()

# Step 5: Modeling (Using ARIMA as an example)
# Splitting data into train and test sets
train_data = stock_data['Close'].iloc[:int(len(stock_data)*0.8)]
test_data = stock_data['Close'].iloc[int(len(stock_data)*0.8):]

# Fitting the ARIMA model
model = ARIMA(train_data, order=(5,1,0))
fitted_model = model.fit()

# Step 6: Model Evaluation
# Evaluate the model on the test data
predictions = fitted_model.forecast(steps=len(test_data))

# Step 7: Forecasting
plt.figure(figsize=(10, 6))
plt.plot(test_data.index, test_data.values, label='Actual Prices', color='blue')
plt.plot(test_data.index, predictions, label='Predicted Prices', color='red')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Step 8: Interpretation and Insights
# You can interpret the results, compare predictions with actual values, and draw insights based on the analysis.
