# **AI-Powered Retail Overview:**
This project utilizes data mining techniques to assist online retail businesses in making informed decisions.

**Data Files:**
customers.csv: Contains information about customers, including their unique IDs, names, gender, age, address details, and location.
sales.csv: Includes sales records, detailing sales IDs, order IDs, product IDs, price per unit, quantity, and total price.

orders.csv: Provides information about customer orders, including order IDs, customer IDs, payment details, order dates, and delivery dates.
products.csv: Holds details about products, including product IDs, types, names, sizes, colors, prices, quantities, and descriptions.

**Python Scripts:**
rfm-analysis.py: Conducts RFM (Recency, Frequency, Monetary) analysis on customer data. It calculates recency, frequency, and monetary values for each customer and segments them into clusters using K-means clustering. The script then prints detailed insights for each cluster, aiding in targeted marketing strategies.

basket-analysis.py: Executes basket analysis on sales data. It identifies patterns of co-occurrence between products purchased by customers and suggests strategies such as cross-selling and upselling.

**Installation**
To run the analysis scripts, follow these steps:
Step 1) Clone this repository to your local machine or download the folder 'data-mining-group27'.
Step 2) Open the terminal at the folder 'data-mining-group27'.
Step 3) Install the required dependencies:
```console
pip install sklearn
pip install pandas
```
Step 4) Run the desired Python script (e.g., python rfm-analysis.py).
