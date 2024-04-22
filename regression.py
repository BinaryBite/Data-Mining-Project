import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('C:\Users\rpyst\Downloads\data-mining-group27-main (1)\data-mining-group27-main\merged_data.csv')
data = data[['product_id', 'price', 'quantity']]
data['price'] = pd.to_numeric(data['price'].str.replace(',', ''))
data['quantity'] = pd.to_numeric(data['quantity'])
X = data[['price']]
y = data['quantity']

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
print("Coefficients: \n", model.coef_)
print("Intercept: \n", model.intercept_)
sns.regplot(x='price', y='quantity', data=data)
plt.show()
