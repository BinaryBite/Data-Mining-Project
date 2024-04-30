import pandas as pd
import itertools
from mlxtend.frequent_patterns import apriori, association_rules

class BasketAnalysis:
    def __init__(self, merged_data_connection, products_connection):
        self.merged_data_connection = merged_data_connection
        self.products_connection = products_connection

    def basket_analysis(self):
        # Load merged data
        full_merged_df = pd.read_csv(self.merged_data_connection)

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

    #This makes sure we are checking the association rules df for all rules that contain combinations of the items in the basket
    def get_combinations(self, lst):
        combinations = []
        for r in range(1, len(lst) + 1):
            combinations.extend(itertools.combinations(lst, r))

        combinations = [frozenset(s) for s in combinations]
        return combinations

    #This recommends the 3 items with the most lift via association rules containing a combination of the ones in the basket
    def recommender(self, basket, rules):
        search_df = pd.DataFrame()
        comb = self.get_combinations(basket)

        for item in comb:
            idx = rules[rules["antecedents"] == item].index.tolist()
            search_df = pd.concat([search_df, rules.loc[idx]], ignore_index=True)

        search_df = search_df.sort_values(by='lift', ascending=False)

        recommended_products = []

        # this is utilized for now as certain consequents contain more than one item (perhaps this can be changed if we propse hypothetically the company sells bundles)
        counter = 0
        for product in search_df["consequents"].values:
            for item in product:
                recommended_products.append(item)
                counter += 1
                if counter == 3:
                    return recommended_products

        return recommended_products

    #This just goes through which item the customer wants via their input, result is currently only product_name as that is what is used in the antecedent and decedents of the association rules
    def driller(self, atts, products_df):
        current = products_df
        for item in atts:
            print(f"What sort of {item} do you want?")
            print("options are {x}".format(x=current[item].unique()))
            i = input()

            if i.lower() not in [x.lower() for x in current[item].unique()]:
                print(f"That's not a valid {item} name.")
                break
            else:
                current = current[current[item].str.lower() == i.lower()]

        result = current["product_name"].iloc[0]
        return result

    def start_basket(self):
        rules_df = self.basket_analysis()
        products_df = pd.read_csv(self.products_connection)

        basket = []
        done = 0

        while done == 0:
            print(f"Your current basket is: {basket}")

            if basket:
                recm = self.recommender(basket, rules_df)
                print(f"We recommend you try: {recm}")

                inp = ""
                while inp != "y" and inp != "n":
                    print("Is that everything? (Y/N)")
                    inp = input(": ")
                    inp = inp.lower()

                if inp == "y":
                    print("Have a nice day!")
                    exit()

            adding = self.driller(["product_type", "product_name", "colour", "size"], products_df)
            basket.append(adding)


if __name__ == "__main__":
    analysis = BasketAnalysis('merged_data.csv', 'products.csv')
    analysis.start_basket()