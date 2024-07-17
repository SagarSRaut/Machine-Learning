import pandas as pd
import matplotlib.pylab as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os

os.chdir('C:\Training\Academy\Statistics (Python)\Association Rules datasets')

fp_df = pd.read_csv('Faceplate.csv',index_col=0)
fp_df.head()

# create frequent itemsets
fp_df = fp_df.astype(bool)
itemsets = apriori(fp_df, min_support=0.2,
                   use_colnames=True)

# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.6)
print(rules[['antecedents','consequents','support',
             'confidence','lift']])

rules.sort_values(by=['lift'], 
                  ascending=False).head(6)

rule_df = rules.sort_values(by=['lift','confidence'], 
                            ascending=False)

print(rule_df[['antecedents','consequents','support',
               'confidence','lift']])

relv_rules = rule_df[rule_df["lift"]>1]

print(relv_rules[['antecedents','consequents',
                  'support','confidence','lift']])

########################################################

fp_df = pd.read_csv('Cosmetics.csv',index_col=0)
fp_df.head()

# create frequent itemsets
fp_df = fp_df.astype(bool)
itemsets = apriori(fp_df, min_support=0.2,
                   use_colnames=True)

# and convert into rules
rules = association_rules(itemsets, metric='confidence', 
                          min_threshold=0.6)
print(rules[['antecedents','consequents','support',
             'confidence','lift']])

rules.sort_values(by=['lift'], 
                  ascending=False)

rule_df = rules.sort_values(by=['lift','confidence'], 
                            ascending=False)

print(rule_df[['antecedents','consequents','support',
               'confidence','lift']])

relv_rules = rule_df[rule_df["lift"]>1]

print(relv_rules[['antecedents','consequents',
                  'support','confidence','lift']])

