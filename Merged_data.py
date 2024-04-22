import pandas as pd

# Define file paths
orders_path = 'orders.csv'
customers_path = 'customers.csv'
products_path = 'products.csv'
sales_path = 'sales.csv'

# Load data
customers_df = pd.read_csv(customers_path)
orders_df = pd.read_csv(orders_path)
products_df = pd.read_csv(products_path)
sales_df = pd.read_csv(sales_path)

# Merge data
sales_orders_df = pd.merge(sales_df, orders_df, on='order_id', how='left')
sales_orders_products_df = pd.merge(sales_orders_df, products_df, left_on='product_id', right_on='product_ID', how='left')
full_data_df = pd.merge(sales_orders_products_df, customers_df, on='customer_id', how='left')

# Display the first few rows of the merged data to check the results
print(full_data_df.head())