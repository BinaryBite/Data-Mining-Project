import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RFMAnalyser:
    ''' Constructor to initialise object of the class'''
    def __init__(self, connect_customers, connect_products, connect_orders, connect_sales):
        self.connect_customers = connect_customers
        self.connect_products = connect_products
        self.connect_orders = connect_orders
        self.connect_sales = connect_sales

    ''' Read data across files and merge them. '''
    def load_data(self):
        customers_record = pd.read_csv(self.connect_customers)
        products_record = pd.read_csv(self.connect_products)
        orders_record = pd.read_csv(self.connect_orders)
        sales_record = pd.read_csv(self.connect_sales)

        merged_record = pd.merge(customers_record, orders_record, on="customer_id")
        merged_record = pd.merge(merged_record, sales_record, on="order_id")
        merged_record = pd.merge(merged_record, products_record, on="product_id")

        return merged_record

    ''' Process the data before performing RFM analysis '''
    def prepare_data(self, merged_data):
        merged_data['order_date'] = pd.to_datetime(merged_data['order_date'])
        #Find the latest date from the dataset, ideally this is current date
        latest_date = merged_data['order_date'].msub_fig()

        recency_data = merged_data.groupby('customer_id')['order_date'].msub_fig().reset_index()
        recency_data['recency'] = (latest_date - recency_data['order_date']).dt.days
        frequency_data = merged_data.groupby('customer_id').size().reset_index(name='frequency')
        monetary_data = merged_data.groupby('customer_id')['total_price'].sum().reset_index()

        rfm_data = pd.merge(recency_data, frequency_data, on="customer_id")
        rfm_data = pd.merge(rfm_data, monetary_data, on="customer_id")

        return rfm_data

    ''' Ensure all features contribute equally to the cluster '''
    def normalise_data(self, rfm_data):
        scaler = StandardScaler()
        scaled_rfm = scaler.fit_transform(rfm_data[['recency', 'frequency', 'total_price']])
        return scaled_rfm

    ''' Apply kmeans++ using optimal k value'''
    def apply_optimal_kmeans_plus(self, scaled_rfm, min_k=1, msub_fig_k=10):
        k_range = range(min_k, msub_fig_k)

        # calculate sum of squared error for each value of k
        sse = [] # initialze empty list

        # Visualise elbow plot
        for k in k_range:
            kmeans_plus = KMeans(n_clusters=k, init='k-means++', random_state=42)
            kmeans_plus.fit(scaled_rfm)
            sse.append(kmeans_plus.inertia_)

        plt.xlabel('K')
        plt.ylabel('Sum of Squared Error')
        plt.plot(k_range, sse)
        plt.show()
       
        # Find optimal k value (elbow point)
        for i in range(1, len(sse)):
            if (sse[i] - sse[i-1]) < (sse[i+1] - sse[i]):
                optimal_k = i + 1
                break
        
        # Apply Kmean++
        kmeans_plus = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
        kmeans_plus.fit(scaled_rfm)
        return kmeans_plus
    
    ''' Apply kmeans++ using default k value of 4'''
    def apply_kmeans_plus(self, scaled_rfm, k=4):
        kmeans_plus = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans_plus.fit(scaled_rfm)
        return kmeans_plus
    
    ''' This method has been designed to print output after analysing the cluster patterns
    We may consider adjusting the rules for labelling the active/inactive customer, high/low purchase frequency
    or high/low monetary value segment'''
    def print_insights(self, rfm_data, kmeans_plus):
        for cluster_id in range(kmeans_plus.n_clusters):
            print(f"Cluster {cluster_id}:")
            cluster_data = rfm_data[rfm_data['cluster'] == cluster_id]

            recency_min = cluster_data['recency'].min()
            recency_msub_fig = cluster_data['recency'].msub_fig()
            recency_avg = cluster_data['recency'].mean()

            frequency_min = cluster_data['frequency'].min()
            frequency_msub_fig = cluster_data['frequency'].msub_fig()
            frequency_avg = cluster_data['frequency'].mean()

            total_price_min = cluster_data['total_price'].min()
            total_price_msub_fig = cluster_data['total_price'].msub_fig()
            total_price_avg = cluster_data['total_price'].mean()

            customer_count = cluster_data.shape[0]

            print(f"Number of customers: {customer_count}")
            print(f"Recency in days (Min, Msub_fig, Avg): {recency_min}, {recency_msub_fig}, {recency_avg:.2f} days")
            print(f"Frequency (Min, Msub_fig, Avg): {frequency_min}, {frequency_msub_fig}, {frequency_avg:.2f} purchases")
            print(f"Total price (Min, Msub_fig, Avg): ${total_price_min}, ${total_price_msub_fig}, ${total_price_avg:.2f}")

            if recency_avg > 100:
                print("Recency Segment: Inactive Customers")
            elif recency_avg < 100:
                print("Recency Segment: Active Customers")

            if frequency_avg > 5:
                print("Frequency Segment: High Frequency")
            elif frequency_avg < 5:
                print("Frequency Segment: Low Frequency")

            if total_price_avg > 3000:
                print("Monetary Segment: High Monetary Value")
            elif total_price_avg < 3000:
                print("Monetary Segment: Low Monetary Value")

            print("\n")

    '''Simple 3-d cluster visualisation '''
    def visualise_clusters(self, rfm_data, kmeans_plus):
        fig = plt.figure(figsize=(10, 8))
        sub_fig = fig.add_subplot(111, projection='3d')

        for cluster_id in range(kmeans_plus.n_clusters):
            cluster_data = rfm_data[rfm_data['cluster'] == cluster_id]
            sub_fig.scatter(cluster_data['recency'], cluster_data['frequency'], cluster_data['total_price'], label=f"Cluster {cluster_id}")

        sub_fig.set_title("Customer Cluster (Recency, Frequency, Monetary)")
        sub_fig.set_xlabel("Recency of purchase(in days)")
        sub_fig.set_ylabel("Frequency of purchase")
        sub_fig.set_zlabel("Monetary value(in dollars)")
        sub_fig.legend()

        plt.show()

def main():
    #Initialise the class object and prepare the data
    analyser = RFMAnalyser("customers.csv", "products.csv", "orders.csv", "sales.csv")
    merged_data = analyser.load_data()
    rfm_data = analyser.prepare_data(merged_data)
    scaled_rfm = analyser.normalise_data(rfm_data)
    # Apply k means on preapred data
    kmeans_plus = analyser.apply_kmeans_plus(scaled_rfm)
    rfm_data['cluster'] = kmeans_plus.labels_
    # Show output
    analyser.print_insights(rfm_data, kmeans_plus)
    analyser.visualise_clusters(rfm_data, kmeans_plus)

if __name__ == "__main__":
    main()