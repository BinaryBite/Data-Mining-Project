import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data from CSV files
customers_df = pd.read_csv("customers.csv")
orders_df = pd.read_csv("orders.csv")
products_df = pd.read_csv("products.csv")
sales_df = pd.read_csv("sales.csv")

# Merge relevant data
merged_df = pd.merge(sales_df, products_df, on="product_id")
merged_df = pd.merge(merged_df, orders_df, on="order_id")
merged_df = pd.merge(merged_df, customers_df, on="customer_id")

# Select relevant features
features = ['gender', 'age', 'city', 'state', 'country',
            'order_date', 'delivery_date',
            'product_type', 'price', 'quantity_y',
            'price_per_unit', 'total_price']

# Filter merged dataframe to include only relevant features
data = merged_df[features]

# Preprocess data (convert date strings to datetime objects, handle missing values, etc.)
data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')
data['delivery_date'] = pd.to_datetime(data['delivery_date'], errors='coerce')

# Drop rows with missing or invalid dates
data.dropna(subset=['order_date', 'delivery_date'], inplace=True)

# Extract relevant features from date columns
data['order_year'] = data['order_date'].dt.year
data['order_month'] = data['order_date'].dt.month
data['order_day'] = data['order_date'].dt.day
data['delivery_year'] = data['delivery_date'].dt.year
data['delivery_month'] = data['delivery_date'].dt.month
data['delivery_day'] = data['delivery_date'].dt.day

# Drop the original date columns
data.drop(columns=['order_date', 'delivery_date'], inplace=True)

# Encode categorical variables (e.g., gender, city, state, country, product_type)
data = pd.get_dummies(data, columns=['gender', 'city', 'state', 'country', 'product_type'])

# Split the data into features (X) and target variable (y)
X = data.drop(columns=['quantity_y'])  # Features
y = data['quantity_y']  # Target variable (product demand)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Make predictions for the entire dataset
predicted_demand = model.predict(X)

# Add predicted demand as a new column in the dataframe
data['predicted_demand'] = predicted_demand

# Group by product types and sum predicted demand for each type
product_demand = (
    data.groupby(['product_type_Jacket', 'product_type_Shirt', 'product_type_Trousers'])
    ['predicted_demand'].sum().reset_index()
)

# Print the demand for each product
print("\nPredicted demand for each product:")
print(product_demand)

# Plot the bar graphs for each product type
plt.figure(figsize=(10, 6))

# Bar graph for product type: Jacket
plt.bar('Jacket', product_demand.loc[product_demand['product_type_Jacket'] == True, 'predicted_demand'], color='skyblue', label='Jacket')

# Bar graph for product type: Shirt
plt.bar('Shirt', product_demand.loc[product_demand['product_type_Shirt'] == True, 'predicted_demand'], color='salmon', label='Shirt')

# Bar graph for product type: Trousers
plt.bar('Trousers', product_demand.loc[product_demand['product_type_Trousers'] == True, 'predicted_demand'], color='lightgreen', label='Trousers')

plt.xlabel('Product Type')
plt.ylabel('Predicted Demand(in million units)')
plt.title('Predicted Demand for Each Product Type')
plt.legend()
plt.tight_layout()
plt.show()

