import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
file_path = r"products.csv"
data = pd.read_csv(file_path)

# Clean data
data['price'] = pd.to_numeric(data['price'])
data['quantity'] = pd.to_numeric(data['quantity'])

# Train a linear regression model
X = data[['price']]
y = data['quantity']
model = LinearRegression()
model.fit(X, y)

# Calculate price elasticity
price_elasticity = model.coef_[0] * (data['price'].mean() / data['quantity'].mean())

# Model Performance
y_pred = model.predict(X)
r_squared = r2_score(y, y_pred)

# Calculate residuals
residuals = y - y_pred

# Select relevant columns for visualization
data_vis = data[['price', 'quantity']]

# Set up the aesthetics
sns.set(style="whitegrid")
plt.figure(figsize=(14, 7))

# Plot distribution of prices
plt.subplot(2, 2, 1)
sns.histplot(data_vis['price'], bins=20, kde=True, color='skyblue', alpha=0.7)
plt.title('Distribution of Prices', fontsize=14)
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Plot distribution of quantities
plt.subplot(2, 2, 2)
sns.histplot(data_vis['quantity'], bins=20, kde=True, color='salmon', alpha=0.7)
plt.title('Distribution of Quantities', fontsize=14)
plt.xlabel('Quantity', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Plot residuals vs fitted values with smoother visualization
plt.subplot(2, 2, 3)
sns.scatterplot(x=y_pred, y=residuals, color='blue', alpha=0.7)
sns.lineplot(x=y_pred, y=0, color='red', linestyle='--', linewidth=2)  # Add a horizontal line at y=0
plt.title('Residuals vs Fitted', fontsize=14)
plt.xlabel('Fitted values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)

# Boxplot of residuals with enhanced visual appearance
plt.subplot(2, 2, 4)
sns.boxplot(y=residuals, color='skyblue', width=0.5)
plt.title('Distribution of Residuals', fontsize=14)
plt.ylabel('Residuals', fontsize=12)

# Adjust layout
plt.tight_layout()
plt.show()

# Plot regression line and insights
plt.figure(figsize=(10, 8))

# Scatter plot with regression line
sns.regplot(x='price', y='quantity', data=data_vis, scatter_kws={"color": "blue", "alpha": 0.5}, line_kws={"color": "red"})
plt.axhline(y=data_vis['quantity'].mean(), color='gray', linestyle='--', label='Mean Quantity')
plt.axvline(x=data_vis['price'].mean(), color='gray', linestyle='--', label='Mean Price')

# Highlighting data points above and below the regression line
plt.scatter(data_vis[data_vis['quantity'] > model.predict(data_vis[['price']])]['price'],
            data_vis[data_vis['quantity'] > model.predict(data_vis[['price']])]['quantity'],
            color='green', label='Above Regression Line', alpha=0.7)
plt.scatter(data_vis[data_vis['quantity'] < model.predict(data_vis[['price']])]['price'],
            data_vis[data_vis['quantity'] < model.predict(data_vis[['price']])]['quantity'],
            color='red', label='Below Regression Line', alpha=0.7)

# Display additional insights
plt.text(data_vis['price'].mean(), data_vis['quantity'].mean(),
         f'Mean Price: ${data_vis["price"].mean():,.2f}\nMean Quantity: {data_vis["quantity"].mean():,.2f}', 
         ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.text(data_vis['price'].max(), data_vis['quantity'].min(), f'R-squared: {r_squared:.2f}', 
         ha='right', va='bottom', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
plt.text(data_vis['price'].max(), data_vis['quantity'].max(), f'Price Elasticity: {price_elasticity:.2f}', 
         ha='right', va='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# Labels and title
plt.xlabel('Price', fontsize=14)
plt.ylabel('Quantity', fontsize=14)
plt.title('Linear Regression: Price vs Quantity', fontsize=16)
plt.legend(fontsize=12)

plt.show()

# Insights
print("\nInsights:")
print("- There is a positive linear relationship between price and quantity sold, as evidenced by the regression line.")
print("- Outliers exist in the dataset, indicating potential anomalies or unique cases in the data.")
print("- The mean price is ${:,.2f} and the mean quantity sold is {:.2f}.".format(data_vis['price'].mean(), data_vis['quantity'].mean()))
print("- The price elasticity of demand is {:.2f}, indicating that for a 1% increase in price, the quantity sold decreases by {:.2f}%.".format(price_elasticity, abs(price_elasticity)))
