import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Load data
customer_data = pd.read_csv('customers.csv')
order_data = pd.read_csv('orders.csv')
sales_data = pd.read_csv('sales.csv')

# Prepare data
order_sales_data = pd.merge(order_data, sales_data, on='order_id', how='left')
merged_data = pd.merge(customer_data, order_sales_data, on='customer_id', how='left')

# Calculate total spend, average order value, and order frequency
order_summary = merged_data.groupby('customer_id').agg(total_spend=('total_price', 'sum'),
                                                        num_orders=('order_id', 'nunique'))
order_summary['avg_order_value'] = order_summary['total_spend'] / order_summary['num_orders']
order_summary.reset_index(inplace=True)

# Merge order summary with combined data
merged_data = pd.merge(merged_data, order_summary, on='customer_id', how='left')

# Calculate order frequency
merged_data['order_frequency'] = merged_data['num_orders']

# Calculate last purchase date for each customer
merged_data['order_date'] = pd.to_datetime(merged_data['order_date'])
last_order_date = merged_data.groupby('customer_id')['order_date'].max()

# Define churn threshold (6 months from November 1st, 2021)
churn_threshold = pd.to_datetime('2021-11-01') - pd.DateOffset(months=6)

# Determine churn status based on last purchase date
merged_data['churn'] = (last_order_date <= churn_threshold).astype(int)

# Drop rows with missing churn labels
merged_data.dropna(subset=['churn'], inplace=True)

# Define features and target variable
features = ['age', 'gender', 'total_spend', 'avg_order_value', 'order_frequency'] 
target = 'churn'  # Target variable indicating whether a customer has churned

# Split data into features and target variable
X = merged_data[features]
y = merged_data[target]

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X)  

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_encoded)

# Train the random forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_imputed, y)

# Predict churn probabilities for all customers
churn_probabilities = classifier.predict_proba(X_imputed)[:, 1]

# Add churn probabilities to the combined data
merged_data['churn_probability'] = churn_probabilities

# Sort customers based on churn probabilities (highest probability of churn first)
sorted_customers = merged_data.sort_values(by='churn_probability', ascending=False)

# Print all customers sorted by churn probabilities
print("Customers sorted by churn probabilities:")
print(sorted_customers[['customer_id', 'customer_name', 'churn_probability']])

plt.figure(figsize=(10, 6))
plt.hist(sorted_customers['churn_probability'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Churn Probability')
plt.xlabel('Churn Probability')
plt.ylabel('Number of customers')
plt.grid(True)
plt.show()