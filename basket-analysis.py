#General Packages
import pandas as pd
import numpy as np

#MBA PAckages
import collections
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

##PRE PREPERATION########################################################################################################################################

#Read in sales.csv
df = pd.read_csv("sales.csv")

#Read in products.csv
df2 = pd.read_csv("products.csv")

#Remove not needed columns
useless = ['sales_id', 'price_per_unit', 'quantity', 'total_price']
df.drop(columns = useless, inplace = True)

#Group all product_ids into rows corresponding to their order_id
gdf = df.groupby('order_id').agg(lambda x: list(x))

#Unpack Lists from grouby
sdf = pd.DataFrame(gdf["product_id"].tolist(), columns = [i for i in range(0, gdf["product_id"].str.len().max())])

#Create a name for each product (the products are not uniquely identifiable by one or two columns alone)
df2["combine"] = df2["colour"] + "-" + df2["size"] + "-" + df2["product_name"]

#Change product_ids to names to help understanding
sdf = sdf.map(lambda x: df2.loc[df2['product_id'] == x, 'combine'].iloc[0] if not pd.isna(x) else None)


##DATA EXPLORATION#####################################################################################################################################

#Overview
d = sdf.describe()

#count of unique values
u = sdf.stack().value_counts()

#count of all values
v = sdf.count(axis=0, numeric_only= False).sum()

#list of purchase lists
t = gdf["product_id"].tolist()

#list of all products bought
products = [i for l in t for i in l if i != None]

##ONE HOT ENCODING####################################################################################################################################

#create the encoder object to fit our list of purchase lists
encoder = TransactionEncoder().fit(t)

#create encoded array
onehot = encoder.transform(t)

#convert array to pd dataframe
onehot_df = pd.DataFrame(onehot, columns = encoder.columns_)

#Give the columns their proper values
col_name_map = {i: name for i, name in enumerate(df2["combine"].values)}
onehot_df = onehot_df.rename(columns = col_name_map)

##RULE DISCOVERY####################################################################################################################################
#Looking into item support
mb_itemset = apriori(onehot_df, min_support = 0.001, use_colnames = True) #processing takes exceptionally long 6 mins at min_threshold = 0.001 final shape:(111706, 2)
itemsup  = mb_itemset.sort_values(by=['support'], ascending = True) 
#itemsup.to_csv("test0")

#Finding the association rules
rules = association_rules(mb_itemset, metric = 'confidence', min_threshold = 0.5 ) #processing takes exceptionally long 1-2 mins at min_threshold = 0.001
#rules.to_csv("association_results.csv", index = False)