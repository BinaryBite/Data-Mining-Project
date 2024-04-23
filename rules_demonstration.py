import pandas as pd
import numpy as np
import itertools
from basket_analysis import ba_func

#This makes sure we are checking the association rules df for all rules that contain combinations of the items in the basket
def get_combinations(lst):
    combinations = []
    for r in range(1, len(lst) + 1):
        combinations.extend(itertools.combinations(lst, r))

    combinations = [frozenset(s) for s in combinations]
    return combinations

#This recommends the 3 items with the most lift via association rules containing a combination of the ones in the basket
def recommender(basket, rules):
    search_df = pd.DataFrame()
    comb = get_combinations(basket)

    for item in comb:

        idx = rules[rules["antecedents"] == item].index.tolist()
        search_df = pd.concat([search_df, rules.loc[idx]], ignore_index=True)

    search_df = search_df.sort_values(by='lift', ascending=False)

    recommended_products = []

    #this is utilized for now as certain consequents contain more than one item (perhaps this can be changed if we propse hypothetically the company sells bundles)
    counter = 0
    for product in search_df["consequents"].values:
        for item in product:
            recommended_products.append(item)
            counter += 1
            if counter == 3:
                return recommended_products
    
    return recommended_products

#This just goes through which item the customer wants via their input, result is currently only product_name as that is what is used in the antecedent and decedents of the association rules
def driller(atts, products_df):
    current = products_df
    for item in atts:
        print(f"What sort of {item} do you want?")
        print("options are {x}".format(x = current[item].unique()))
        i = input()

        if i not in current[item].unique():
                    print(f"That's not a valid {item} name.")
                    break
        else:
            current = current[current[item] == i]
    
    result = current["product_name"].iloc[0]
    return result


#Main program
def start_basket(rules_df, products_df):

    basket = []
    done = 0

    while done == 0:
        print(f"Your current basket is: {basket}")
        

        if basket:

            recm = recommender(basket, rules_df)

            print(f"We recommend you try: {recm}")

            inp = ""
            while inp != "y" and inp != "n":

                print(inp)
                print("Is that everything? (Y/N)")
                inp = input(": ")
                inp = inp.lower()
            
            if inp == "y":
                print("Have a nice day!")
                exit()

        adding = driller(["product_type","product_name", "colour", "size"], products_df)
        basket.append(adding)
                    
#loading rules directly into this program from Lucy's mba as content is lost on conversion to csv            
r = ba_func()
p = pd.read_csv("products.csv")
start_basket(r,p)