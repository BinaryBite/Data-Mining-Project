import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data from CSV files
customers_df = pd.read_csv("customers.csv")
orders_df = pd.read_csv("orders.csv")
products_df = pd.read_csv("products.csv")
sales_df = pd.read_csv("sales.csv")

# Merge relevant data
merged_df = pd.merge(customers_df, orders_df, on="customer_id")
merged_df = pd.merge(merged_df, sales_df, on="order_id")
merged_df = pd.merge(merged_df, products_df, on="product_id")

# Calculate Recency, Frequency, Monetary values
merged_df['order_date'] = pd.to_datetime(merged_df['order_date'])
recency_df = merged_df.groupby('customer_id')['order_date'].max().reset_index()
recency_df['recency'] = (pd.to_datetime('now') - recency_df['order_date']).dt.days

frequency_df = merged_df.groupby('customer_id').size().reset_index(name='frequency')

monetary_df = merged_df.groupby('customer_id')['total_price'].sum().reset_index()

# Merge RFM values
rfm_df = pd.merge(recency_df, frequency_df, on="customer_id")
rfm_df = pd.merge(rfm_df, monetary_df, on="customer_id")

# Standardize the features
scaler = StandardScaler()
scaled_rfm = scaler.fit_transform(rfm_df[['recency', 'frequency', 'total_price']])

# Apply K-means clustering with K-means++ initialization
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(scaled_rfm)

# Assign clusters to customers
rfm_df['cluster'] = kmeans.labels_

# Print insights for each cluster
for cluster_id in range(kmeans.n_clusters):
    print(f"Cluster {cluster_id}:")
    cluster_data = rfm_df[rfm_df['cluster'] == cluster_id]
    
    # Calculate statistics
    recency_min = cluster_data['recency'].min()
    recency_max = cluster_data['recency'].max()
    recency_avg = cluster_data['recency'].mean()
    
    frequency_min = cluster_data['frequency'].min()
    frequency_max = cluster_data['frequency'].max()
    frequency_avg = cluster_data['frequency'].mean()
    
    total_price_min = cluster_data['total_price'].min()
    total_price_max = cluster_data['total_price'].max()
    total_price_avg = cluster_data['total_price'].mean()
    
    customer_count = cluster_data.shape[0]
    
    # Print insights
    print(f"Number of customers: {customer_count}")
    print(f"Recency (Min, Max, Avg): {recency_min}, {recency_max}, {recency_avg:.2f} days")
    print(f"Frequency (Min, Max, Avg): {frequency_min}, {frequency_max}, {frequency_avg:.2f} purchases")
    print(f"Total price (Min, Max, Avg): ${total_price_min}, ${total_price_max}, ${total_price_avg:.2f}")
    
    # Additional insights based on RFM analysis
    if recency_avg > 900:
        print("Recency Segment: Inactive Customers")
    elif recency_avg < 600:
        print("Recency Segment: Active Customers")
        
    if frequency_avg > 10:
        print("Frequency Segment: High Frequency")
    elif frequency_avg < 10:
        print("Frequency Segment: Low Frequency")
        
    if total_price_avg > 1500:
        print("Monetary Segment: High Monetary Value")
    elif total_price_avg < 1500:
        print("Monetary Segment: Low Monetary Value")
        
    print("\n")
