import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk

# Load sales and products data from CSV files
sales_df = pd.read_csv("data-mining-group27-main/sales.csv")
products_df = pd.read_csv("data-mining-group27-main/products.csv")

# Merge sales data with products data
merged_df = pd.merge(sales_df, products_df, on="product_id")

# Aggregate total sales (total price) for each product
product_sales = merged_df.groupby('product_id').agg({'total_price': 'sum'}).reset_index()

# Merge sales data with aggregated product sales
products_df = pd.merge(products_df, product_sales, on="product_id", how="left")

# Calculate total sales for each product
product_sales_total = products_df.groupby('product_name')['total_price'].sum().reset_index()

# Define a threshold for popular products (e.g., top 20%)
popular_threshold = product_sales_total['total_price'].quantile(0.8)

# Create a binary label for popular products
product_sales_total['popular'] = (product_sales_total['total_price'] >= popular_threshold).astype(int)

# Merge the popularity label with the products dataframe
products_df = pd.merge(products_df, product_sales_total[['product_name', 'popular']], on="product_name", how="left")

# Drop duplicate products
products_df.drop_duplicates(subset='product_name', inplace=True)

def check_popularity(selected_products):
    # Filter products dataframe based on selected products
    selected_products_df = products_df[products_df['product_name'].str.lower().isin(p.lower() for p in selected_products)]
    
    if len(selected_products_df) == 0:
        raise ValueError("No data available for the selected products.")
    
    # Define features and target variable
    X = selected_products_df[['price', 'total_price']]
    y = selected_products_df['popular']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize and train the random forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    # Add predicted popularity as a new column in the dataframe
    selected_products_df['predicted_popularity'] = model.predict(X_scaled)
    
    return selected_products_df

def visualize_popularity(selected_products):
    try:
        # Check popularity of selected products
        selected_products_df = check_popularity(selected_products)

        # Plot popularity based on total sales
        plt.figure(figsize=(12, 8))
        sns.barplot(x='product_name', y='total_price', hue='predicted_popularity', data=selected_products_df,
                    palette=['#4CAF50', '#F44336'], edgecolor='gray')
        plt.title('Popularity Based on Total Sales', fontsize=16)
        plt.xlabel('Product', fontsize=14)
        plt.ylabel('Total Sales', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Predicted Popularity', fontsize=12, title_fontsize=12)
        plt.tight_layout()
        plt.show()
    except ValueError as e:
        print("Error:", e)

def on_button_click():
    selected_products = [listbox.get(idx) for idx in listbox.curselection()]
    visualize_popularity(selected_products)

# Create Tkinter window
window = tk.Tk()
window.title("Product Popularity Visualization")

# Create a listbox to display product names
listbox = tk.Listbox(window, selectmode=tk.MULTIPLE)
for product in products_df['product_name']:
    listbox.insert(tk.END, product)
listbox.pack(pady=10)

# Create a button to trigger visualization
button = ttk.Button(window, text="Visualize", command=on_button_click)
button.pack(pady=5)

# Run the Tkinter event loop
window.mainloop()
