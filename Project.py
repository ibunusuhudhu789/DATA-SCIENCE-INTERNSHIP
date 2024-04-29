# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('Hackathon_Ideal_Data.csv')

# Data cleaning
# Drop duplicates
data.drop_duplicates(inplace=True)

# Handle missing values
data.dropna(inplace=True)

# Data analysis
# Summary statistics
summary_stats = data.describe()

# Correlation analysis
correlation_matrix = data.corr()

# Data visualization
# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(data['column'], bins=20, kde=True)
plt.title('Distribution of Column Data')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='column_x', y='column_y', data=data)
plt.title('Scatter Plot between Column X and Column Y')
plt.xlabel('Column X')
plt.ylabel('Column Y')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()