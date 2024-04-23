AI-Powered Retail
Overview
This project analyzes uses datan mining techniques to help online retail business.

Data Files
customers.csv
This file contains information about customers, including their unique IDs, names, gender, age, address details, and location.

sales.csv
This file contains sales records, including details such as sales IDs, order IDs, product IDs, price per unit, quantity, and total price.

orders.csv
This file contains information about customer orders, including order IDs, customer IDs, payment details, order dates, and delivery dates.

products.csv
This file contains details about products, including product IDs, types, names, sizes, colors, prices, quantities, and descriptions.

Python Scripts
rfm-analysis.py
This script performs RFM (Recency, Frequency, Monetary) analysis on customer data. It calculates recency, frequency, and monetary values for each customer and segments them into clusters using K-means clustering. It then prints detailed insights for each cluster, helping in targeted marketing strategies.

basket-analysis.py
This script performs basket analysis on sales data. It identifies patterns of co-occurrence between products purchased by customers and suggests strategies such as cross-selling and upselling.

Installation
To run the analysis scripts, follow these steps:

1)Clone this repository to your local machine or download the folder 'data-mining-group27'.
2)Open terminal at folder 'data-mining-group27'
3)Install the required dependencies:
pip install sklearn
pip install pandas
4)Run the desired Python script (e.g., python rfm-analysis.py).
