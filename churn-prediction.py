import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load customer data
customer_data = pd.read_csv('customers.csv')

# Load order data
order_data = pd.read_csv('orders.csv')

# Load sales data
sales_data = pd.read_csv('sales.csv')

# Merge order and sales data
order_sales_data = pd.merge(order_data, sales_data, on='order_id', how='left')

# Merge customer data with order and sales data
combined_data = pd.merge(customer_data, order_sales_data, on='customer_id', how='left')

# Calculate total spend, average order value, and order frequency
order_summary = combined_data.groupby('customer_id').agg(total_spend=('total_price', 'sum'),
                                                        num_orders=('order_id', 'nunique'))
order_summary['avg_order_value'] = order_summary['total_spend'] / order_summary['num_orders']
order_summary.reset_index(inplace=True)

# Merge order summary with combined data
combined_data = pd.merge(combined_data, order_summary, on='customer_id', how='left')

# Calculate order frequency (assuming 1 order per row in the order data)
combined_data['order_frequency'] = combined_data['num_orders']

# Calculate last purchase date for each customer
combined_data['order_date'] = pd.to_datetime(combined_data['order_date'])
last_purchase_date = combined_data.groupby('customer_id')['order_date'].max()

# Define churn threshold (6 months from November 1st, 2021)
churn_threshold = pd.to_datetime('2021-11-01') - pd.DateOffset(months=6)

# Determine churn status based on last purchase date
combined_data['churn'] = (last_purchase_date <= churn_threshold).astype(int)

# Drop rows with missing churn labels
combined_data.dropna(subset=['churn'], inplace=True)

# Define features and target variable
features = ['age', 'gender', 'total_spend', 'avg_order_value', 'order_frequency']  # Adjust feature list as needed
target = 'churn'  # Target variable indicating whether a customer has churned

# Split data into features and target variable
X = combined_data[features]
y = combined_data[target]

# Encode categorical variables (if needed)
X_encoded = pd.get_dummies(X)  # One-hot encoding for categorical variables

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_encoded)

# Train the random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_imputed, y)

# Predict churn probabilities for all customers
churn_probabilities = rf_classifier.predict_proba(X_imputed)[:, 1]

# Add churn probabilities to the combined data
combined_data['churn_probability'] = churn_probabilities

# Sort customers based on churn probabilities (highest probability of churn first)
sorted_customers = combined_data.sort_values(by='churn_probability', ascending=False)

# Print all customers sorted by churn probabilities
print("Customers sorted by churn probabilities:")
print(sorted_customers[['customer_id', 'customer_name', 'churn_probability']])

# Assuming 'sorted_customers' DataFrame contains 'churn_probability' column
plt.figure(figsize=(10, 6))
plt.hist(sorted_customers['churn_probability'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Churn Probability')
plt.xlabel('Churn Probability')
plt.ylabel('Number of customers')
plt.grid(True)
plt.show()
