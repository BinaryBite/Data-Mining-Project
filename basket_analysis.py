import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def ba_func():
    # Load merged data
    full_merged_df = pd.read_csv('merged_data.csv')

    # Prepare data for market basket analysis
    basket_sets = full_merged_df.groupby(['order_id', 'product_name'])['quantity_x'].sum().unstack().reset_index().fillna(0).set_index('order_id')
    basket_sets = (basket_sets > 0).astype(int)  # Ensure boolean type usage

    # Use Apriori algorithm to find frequent itemsets with a minimum support of 0.01
    frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)

    # Generate association rules from frequent itemsets using lift as the metric, minimum lift set to 1
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Sort association rules by lift, confidence, and support in descending order
    sorted_rules = rules.sort_values(by=['lift', 'confidence', 'support'], ascending=False)

    return sorted_rules
    # Save the sorted association rules to a new CSV file
    
result = ba_func()
result.to_csv('sorted_association_rules.csv', index=False)
print("Association rules have been processed and saved successfully.")
