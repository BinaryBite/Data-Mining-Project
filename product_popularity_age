import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load data
merged_df = pd.read_csv("data-mining-group27-main\merged_data.csv") #replace this with the merged.csv

# Preprocess data
# Assuming 'age' column contains age groups
# Encode age groups
label_encoder = LabelEncoder()
merged_df['age_group_encoded'] = label_encoder.fit_transform(merged_df['age'])

# Define age group categories
age_groups = {
    'Kids (0-20)': list(range(0, 21)),
    'Adults (21-60)': list(range(21, 61)),
    'Senior Citizens (61+)': list(range(61, merged_df['age_group_encoded'].max() + 1))
}

# Create a dictionary to map encoded age group to age group labels
age_labels = {code: label for code, label in zip(merged_df['age_group_encoded'], merged_df['age'])}

# Function to train model and predict popular products based on age group
def predict_popular_products():
    # Extract features and target variable
    X = merged_df[['age_group_encoded']]
    y = merged_df['product_name']
    
    # Train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

# Function to visualize popular products for selected age groups
def visualize_popular_products(age_group):
    # Filter data for selected age groups
    selected_data = merged_df[merged_df['age_group_encoded'].isin(age_group)]
    
    if len(selected_data) == 0:
        messagebox.showinfo("Error", "No data available for selected age groups.")
        return
    
    # Predict popular products
    model = predict_popular_products()
    popular_products = model.predict(selected_data[['age_group_encoded']])
    
    # Get the age group labels for display
    age_group_labels = [age_labels[age_group[0]]] * len(selected_data['product_name'].unique())

    # Plot popular products
    plt.figure(figsize=(12, 6))
    plt.hist(popular_products, bins=len(selected_data['product_name'].unique()))
    plt.xlabel('Age Group - Product')
    plt.ylabel('Frequency')
    plt.title('Popular Products for Selected Age Groups')
    plt.xticks(rotation=45, ha='right')
    
    # Add age group labels to the x-axis
    plt.gca().set_xticklabels([f"{age_group_label}\n{product}" for age_group_label, product in zip(age_group_labels, selected_data['product_name'].unique())])
    
    plt.tight_layout()
    plt.show()

# Function to handle button click event
def on_button_click():
    age_group_category = age_group_combobox.get()
    age_group = age_groups[age_group_category]
    visualize_popular_products(age_group)

# Create Tkinter window
window = tk.Tk()
window.title("Product Popularity Prediction")

# Create label and combobox for selecting age group category
age_label = ttk.Label(window, text="Select Age Group Category:")
age_label.grid(row=0, column=0, padx=10, pady=5)
age_group_combobox = ttk.Combobox(window, values=list(age_groups.keys()))
age_group_combobox.grid(row=0, column=1, padx=10, pady=5)

# Create button to trigger visualization
button = ttk.Button(window, text="Visualize", command=on_button_click)
button.grid(row=1, column=0, columnspan=2, padx=10, pady=5)

# Run the Tkinter event loop
window.mainloop()
