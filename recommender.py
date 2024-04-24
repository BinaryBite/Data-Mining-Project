import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
customers = pd.read_csv('customers.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
sales = pd.read_csv('sales.csv')

# Prepare data
merged_data = pd.merge(customers, orders, on='customer_id')
merged_data = pd.merge(merged_data, sales, on='order_id')
merged_data = pd.merge(merged_data, products, on='product_id')

# Define age groups
age_brackets = [0, 20, 30, 40, 50, 60, 100]
age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61+']
merged_data['age_group'] = pd.cut(merged_data['age'], bins=age_brackets, labels=age_labels)

# Segmentation features
features = ['price_per_unit', 'quantity_x']

# Normalise the features(We can consider using MinMaxScaler from class examples?)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data[features])

# Use KMeans for segmentation(can we use Kmeans++ ?)
kmeans = KMeans(n_clusters=5, random_state=42)
merged_data['segment'] = kmeans.fit_predict(scaled_data)

# Analyse purchase patterns by region-age segment
segment = merged_data.groupby(['state', 'age_group', 'product_name']).size().reset_index(name='purchase_count')

# Find the most purchased product within each segment
segment = segment.sort_values(by=['state', 'age_group', 'purchase_count'], ascending=[True, True, False])
segment = segment.groupby(['state', 'age_group']).first().reset_index()

# Display popular products by segment 
print("Popular products by region and age group:")
print(segment)