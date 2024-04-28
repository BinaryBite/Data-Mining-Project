# **AI-Powered Retail Overview:**
This project utilizes data mining techniques to assist online retail businesses in making informed decisions.

**Data Files:**
1) customers.csv: Contains information about customers, including their unique IDs, names, gender, age, address details, and location.

2) sales.csv: Includes sales records, detailing sales IDs, order IDs, product IDs, price per unit, quantity, and total price.

3) orders.csv: Provides information about customer orders, including order IDs, customer IDs, payment details, order dates, and delivery dates.

4) products.csv: Holds details about products, including product IDs, types, names, sizes, colors, prices, quantities, and descriptions.

**Python Scripts:**
1) rfm-analysis.ipynb: Conducts RFM (Recency, Frequency, Monetary) analysis on customer data. It calculates recency, frequency, and monetary values for each customer and segments them into clusters using K-means++clustering. The script then prints detailed insights for each cluster, aiding in targeted marketing strategies.

2) basket-analysis.ipynb: Executes basket analysis on sales data. It identifies patterns of co-occurrence between products purchased by customers. This feature cam help online retail business in implementing strategies such as cross-selling and upselling.

3) recommender.ipynb: Recommends products to customers who share similar taste. For demonstartion purpose, use rating has been used to analyse similarity.

4) sales_prediction.ipynb : Predicts monthly sales for 2022 by taking into account the sale for 2021 for both online and offline purchase mode.

**Installation**
IMPORTANT Pre-requisite : Python, Jupyter Notebook

To run the analysis scripts, follow these steps:
Step 1) Download the folder 'data-mining-group27'.
Step 2) Open the terminal at the folder 'data-mining-group27'.
Step 3) Install the required dependencies:
```console
pip install pandas
pip install sklearn
pip install mlxtend
pip install numpy
pip install matplotlib
pip install lenskit
```
Step 4) Run the desired Python scripts

**References**

Data Set : Bhatia, R. (2020) ‘Shopping Cart Database’, Kaggle. Available at:  https://www.kaggle.com/datasets/ruchi798/shopping-cart-database?select=sales.csv (Accessed April 2024)

RFM Analysis: Hanna, K.T. and Wright, G. (2024) What is RFM analysis (recency, Frequency, monetary)?: Definition from TechTarget, Data Management. Available at: https://www.techtarget.com/searchdatamanagement/definition/RFM-analysis (Accessed: April 2024)

Random Forest Regressor: Random Forest regression | python (2021) YouTube. Available at: https://www.youtube.com/watch?v=jkOtBYZ86Os&t=497s (Accessed: April 2024).

Recommender System: Yarosh, S. (2024) ‘Recommender Systems Lab: Crash Coursh AI #16’, Google Colab. Available at: https://colab.research.google.com/drive/1-v9cw18wTDjaCUlECKHsQnHeisLKyG8U (Accessed April 2024)