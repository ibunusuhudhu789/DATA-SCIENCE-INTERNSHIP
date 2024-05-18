import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
dates = pd.date_range(start='1920-01-01', end='2020-12-31', freq='D')

# Generate synthetic temperature data (in Celsius)
temperature = 10 + 15 * np.sin(np.linspace(0, 3 * np.pi, len(dates))) + np.random.normal(0, 3, len(dates))

# Generate synthetic precipitation data (in mm)
precipitation = np.random.poisson(5, len(dates))

# Generate synthetic humidity data (in %)
humidity = 50 + 10 * np.random.normal(0, 1, len(dates))

# Create DataFrame
data = pd.DataFrame({
    'date': dates,
    'temperature': temperature,
    'precipitation': precipitation,
    'humidity': humidity
})

# Save to CSV
data.to_csv('synthetic_climate_data.csv', index=False)

# Display first few rows
print(data.head())

# Plot to visualize
plt.figure(figsize=(14, 7))
plt.plot(data['date'], data['temperature'], label='Temperature')
plt.title('Synthetic Temperature Data Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
# Import necessary libraries
import pandas as pd

# Load the synthetic dataset
df = pd.read_csv('synthetic_climate_data.csv')

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Display the first few rows to confirm loading
print(df.head())
# Handling missing values
df.fillna(method='ffill', inplace=True)  # Forward fill for simplicity

# Checking for and removing outliers if necessary
df = df[(df['temperature'] > df['temperature'].quantile(0.01)) & (df['temperature'] < df['temperature'].quantile(0.99))]
# Summary statistics
print(df.describe())

# Histogram for temperature
plt.figure(figsize=(10, 5))
sns.histplot(df['temperature'], kde=True)
plt.title('Temperature Distribution')
plt.show()

# Time series plot for temperature
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['temperature'], label='Temperature')
plt.title('Temperature Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
# Moving Average
df['temp_moving_avg'] = df['temperature'].rolling(window=365).mean()

# Decomposition
result = seasonal_decompose(df['temperature'], model='additive', period=365)
result.plot()
plt.show()

# Mann-Kendall Trend Test (using a simplified version)
import pymannkendall as mk
result = mk.original_test(df['temperature'])
print(result)

# Linear Regression for Trend
from sklearn.linear_model import LinearRegression
X = np.array(df.index.year).reshape(-1, 1)
y = df['temperature'].values
model = LinearRegression().fit(X, y)
trend = model.predict(X)

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['temperature'], label='Temperature')
plt.plot(df.index, trend, label='Trend', color='red')
plt.title('Temperature Trend Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
import plotly.express as px

# Interactive line plot
fig = px.line(df, x=df.index, y='temperature', title='Interactive Temperature Over Time')
fig.show()

# Heatmap for temperature by year and month
df['year'] = df.index.year
df['month'] = df.index.month
heatmap_data = df.pivot_table(values='temperature', index='year', columns='month', aggfunc=np.mean)
fig = px.imshow(heatmap_data, labels=dict(x="Month", y="Year", color="Temperature"),
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                title='Temperature Heatmap')
fig.show()
