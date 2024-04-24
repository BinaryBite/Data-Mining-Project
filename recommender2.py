import pandas as pd
import numpy as np
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser

def merge_data():
    customers = pd.read_csv('customers.csv')
    orders = pd.read_csv('orders.csv')
    products = pd.read_csv('products.csv')
    sales = pd.read_csv('sales.csv')

    # Prepare data
    merged_data = pd.merge(customers, orders, on='customer_id')
    merged_data = pd.merge(merged_data, sales, on='order_id')
    merged_data = pd.merge(merged_data, products, on='product_id')
    
    return merged_data

def add_age(data):
    age_brackets = [0, 20, 30, 40, 50, 60, 100]
    age_labels = ['0-20', '21-30', '31-40', '41-50', '51-60', '61+']
    data['age_group'] = pd.cut(data['age'], bins=age_brackets, labels=age_labels)

    return data

def add_ratings(data):
    data['rating'] = np.random.randint(1, 6, size = data.shape[0])

    return data

def recommender2(user_id):
    # Load data
    merged_data = merge_data()
    
    # Define age groups
    merged_data = add_age(merged_data)

    # Create ratings
    rating_data = add_ratings(merged_data)

    #Represent ratings
    rating_data = rating_data[['customer_id','product_name','rating']] #removed age
    rating_data.rename(columns = {'customer_id' : 'user', 'product_name': 'item'}, inplace= True)


    user_user = UserUser(15, min_nbrs = 3)
    algo = Recommender.adapt(user_user)
    algo.fit(rating_data)

    user_recs = algo.recommend(user_id, 3,)
    print(user_recs)


recommender2()