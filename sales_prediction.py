import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class SalesPredictor:
    def __init__(self):
        self.sales_data = None
        self.orders_data = None
        self.merged_data = None
        self.monthly_sales = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.predicted_sales = None

    def import_data(self, sales, orders):
        # Load data from CSV files
        self.sales_data = pd.read_csv(sales)
        self.orders_data = pd.read_csv(orders)

    def prepare_data(self):
        # Merge sales data with orders data based on order ID
        self.merged_data = pd.merge(self.sales_data, self.orders_data, on='order_id')

        # Extract month and year from order date
        self.merged_data['order_date'] = pd.to_datetime(self.merged_data['order_date'])
        self.merged_data['year'] = self.merged_data['order_date'].dt.year
        self.merged_data['month'] = self.merged_data['order_date'].dt.month

        # Calculate monthly sales
        self.monthly_sales = self.merged_data.groupby(['year', 'month']).agg({'total_price': 'sum'}).reset_index()

        # Add a new column for online purchase (randomly generate 0 or 1) 
        self.monthly_sales['online_purchase'] = np.random.randint(0, 2, size=len(self.monthly_sales))

    def apply_rf(self):
        self.X = self.monthly_sales[['year', 'month', 'online_purchase']]
        self.y = self.monthly_sales['total_price']

        # Split data into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=42)
        
        self.rf_model = RandomForestRegressor()
        self.rf_model.fit(self.X_train, self.y_train)
        # Make predictions for the next 12 months
        future_df = pd.DataFrame({'year': [2022] * 12, 'month': range(1, 13), 'online_purchase': [0] * 12})
        self.predicted_sales = self.rf_model.predict(future_df)

    def visualise(self):
        # Plot actual and predicted sales for both online and offline purchases
        fig, graph = plt.subplots(2, figsize=(10, 10))
        
        # Plot for offline purchases
        offline_sales = self.monthly_sales[self.monthly_sales['online_purchase'] == 0]
        graph[0].plot(offline_sales.index[-12:], offline_sales['total_price'].tail(12), marker='o', color='green', label='Actual Offline Sales 2021')
        graph[0].plot(range(12), self.predicted_sales, marker='x', color='blue', linestyle='--', label='Predicted Offline Sales 2022')
        graph[0].set_title('Offline Sales')
        graph[0].set_xlabel('Month')
        graph[0].set_ylabel('Total Sales (in dollars)')
        graph[0].legend()
        
        # Plot for online purchases
        online_sales = self.monthly_sales[self.monthly_sales['online_purchase'] == 1]
        graph[1].plot(online_sales.index[-12:], online_sales['total_price'].tail(12), marker='o', color='green', label='Actual Online Sales 2021')
        graph[1].plot(range(12), self.predicted_sales, marker='x', color='blue', linestyle='--', label='Predicted Online Sales 2022')
        graph[1].set_title('Online Sales')
        graph[1].set_xlabel('Month')
        graph[1].set_ylabel('Total Sales (in dollars)')
        graph[1].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Create an instance of the SalesPredictor class
    predictor = SalesPredictor()
    predictor.import_data('sales.csv', 'orders.csv')
    predictor.prepare_data()

    # Apply random forest regression on the data set
    predictor.apply_rf()

    # Visualise the result
    predictor.visualise()

if __name__ == "__main__":
    main()
