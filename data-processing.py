import csv
import random
from datetime import datetime, timedelta

# Function to generate random date within a specified range
def random_date(start_date, end_date):
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)

# Define start and end dates for January 2024 to April 22, 2024
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 4, 22)

# Load existing customer IDs from customers.csv
existing_customer_ids = []

with open('customers.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        existing_customer_ids.append(int(row['customer_id']))

# Load existing order IDs from orders.csv
existing_order_ids = []

with open('orders.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        existing_order_ids.append(int(row['order_id']))

# Load existing sales IDs from sales.csv
existing_sales_ids = []

with open('sales.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        existing_sales_ids.append(int(row['sales_id']))

# Start order_id from the last order_id plus 1
start_order_id = max(existing_order_ids) + 1 if existing_order_ids else 1

# Start sales_id from the last sales_id plus 1
start_sales_id = max(existing_sales_ids) + 1 if existing_sales_ids else 1

# Load existing product IDs and prices from products.csv
product_prices = {}

with open('products.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        product_id = int(row['product_id'])
        price = float(row['price'])
        product_prices[product_id] = price

# Generate 500 new rows for orders.csv
new_orders = []
for i in range(start_order_id, start_order_id + 200):
    # Pick a random customer ID from the existing ones
    customer_id = random.choice(existing_customer_ids)
    
    # Generate random order date between January 2024 to April 22, 2024
    order_date = random_date(start_date, end_date)
    
    # Generate random delivery date after the order date
    delivery_date = order_date + timedelta(days=random.randint(1, 14))
    
    # Append new order row to the list
    new_orders.append({
        'order_id': i,
        'customer_id': customer_id,
        'payment': random.randint(10000, 99999),
        'order_date': order_date.strftime('%Y-%m-%d'),
        'delivery_date': delivery_date.strftime('%Y-%m-%d')
    })

# Append the new rows to orders.csv
with open('orders.csv', 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['order_id', 'customer_id', 'payment', 'order_date', 'delivery_date'])
    if file.tell() == 0:  # Check if the file is empty
        writer.writeheader()  # Write header if the file is empty
    for row in new_orders:
        writer.writerow(row)

# Generate corresponding sales records only for newly added orders
new_sales = []
for order in new_orders:
    order_id = order['order_id']
    for i in range(start_sales_id, start_sales_id + random.randint(1, 3)):
        # Pick a random product ID from the existing ones
        product_id = random.choice(list(product_prices.keys()))
        
        # Retrieve the price for the selected product ID
        price_per_unit = product_prices[product_id]
        
        # Generate random quantity
        quantity = random.randint(1, 3)
        
        # Append new sales row to the list
        new_sales.append({
            'sales_id': i,
            'order_id': order_id,
            'product_id': product_id,
            'price_per_unit': price_per_unit,
            'quantity': quantity,
            'total_price': price_per_unit * quantity
        })

# Append the new rows to sales.csv
with open('sales.csv', 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['sales_id', 'order_id', 'product_id', 'price_per_unit', 'quantity', 'total_price'])
    if file.tell() == 0:  # Check if the file is empty
        writer.writeheader()  # Write header if the file is empty
    for row in new_sales:
        writer.writerow(row)

print("500 new orders have been added to orders.csv, and corresponding sales records have been added to sales.csv.")
