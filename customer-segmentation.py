import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV files
customer_data = pd.read_csv('customers.csv')
order_data = pd.read_csv('orders.csv')
product_data = pd.read_csv('products.csv')
sales_data = pd.read_csv('sales.csv')

# Merge relevant data
merged_data = pd.merge(customer_data, order_data, on='customer_id')
merged_data = pd.merge(merged_data, sales_data, on='order_id')
merged_data = pd.merge(merged_data, product_data, on='product_id')

# Preprocess data
# Handle missing values
merged_data.dropna(inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
merged_data['gender'] = label_encoder.fit_transform(merged_data['gender'])
print(merged_data.columns)
# Feature Engineering
# Add additional features if needed

# Select relevant features
X = merged_data[['age', 'gender', 'price', 'quantity_y']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply clustering algorithms ( K-means)
kmeans = KMeans(n_clusters=4, random_state=42)
merged_data['cluster'] = kmeans.fit_predict(X_scaled)

# Apply classification algorithms (Random Forest)
# Define target variable
y = merged_data['cluster']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict clusters on test data
y_pred = rf_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Use insights from clustering and classification to drive online sales strategies
# Print unique customer segments
print("Customer Segments:")
print(merged_data['cluster'].unique())

# Group data by cluster
cluster_groups = merged_data.groupby('cluster')

# Print characteristics of each segment
for cluster_id, cluster_data in cluster_groups:
    print(f"Segment {cluster_id}:")
    print("Number of customers:", len(cluster_data))
    print("Age (mean):", cluster_data['age'].mean())
    print("Gender distribution:", cluster_data['gender'].value_counts())
    print("Price (mean):", cluster_data['price'].mean())
    print("Quantity (mean):", cluster_data['quantity_y'].mean())
    print()
