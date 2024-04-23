import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data from CSV files
products_df = pd.read_csv("products.csv")
sales_df = pd.read_csv("sales.csv")

# Merge sales data with products data
merged_df = pd.merge(sales_df, products_df, on="product_id")

# Calculate total quantity sold for each product
product_sales = merged_df.groupby('product_id')['quantity_y'].sum().reset_index()

# Define a threshold for popular products (e.g., top 20%)
popular_threshold = product_sales['quantity_y'].quantile(0.8)

# Create a binary label for popular products
product_sales['popular'] = (product_sales['quantity_y'] >= popular_threshold).astype(int)

# Merge the popular label with the products dataframe
products_df = pd.merge(products_df, product_sales[['product_id', 'popular']], on="product_id", how="left")

# Drop NaN values (products without sales)
products_df.dropna(subset=['popular'], inplace=True)

# Define features and target variable
X = products_df[['price', 'quantity']]
y = products_df['popular']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Add predicted popularity as a new column in the dataframe
products_df['predicted_popularity'] = model.predict(scaler.transform(products_df[['price', 'quantity']]))

# Filter the dataframe to include only popular products
popular_products = products_df[products_df['predicted_popularity'] == 1]

# Print all popular products with their names and predicted popularity scores
print(popular_products[['product_id', 'product_name', 'predicted_popularity']])