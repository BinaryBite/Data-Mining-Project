import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data 
products = pd.read_csv("products.csv")
sales = pd.read_csv("sales.csv")

# Prepare data
merged_data = pd.merge(sales, products, on="product_id")

# Calculate total quantity sold for each product
product_sales = merged_data.groupby('product_id')['quantity_y'].sum().reset_index()

# Define a threshold for popular products (e.g., top 20%)
popular_threshold = product_sales['quantity_y'].quantile(0.8)

# Create a binary label for popular products
product_sales['popular'] = (product_sales['quantity_y'] >= popular_threshold).astype(int)

# Merge the popular label with the products dataframe
products = pd.merge(products, product_sales[['product_id', 'popular']], on="product_id", how="left")

# Drop NaN values (products without sales)
products.dropna(subset=['popular'], inplace=True)

X = products[['price', 'quantity']] # Feature
y = products['popular'] # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalise the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classifier
classification_model = RandomForestClassifier(n_estimators=100, random_state=42)
classification_model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = classification_model.predict(X_test_scaled)

# Add predicted popularity as a new column in the data
products['predicted_popularity'] = classification_model.predict(scaler.transform(products[['price', 'quantity']]))

# Print all popular products with their names and predicted popularity scores
print(products[['product_id', 'product_name', 'predicted_popularity']])