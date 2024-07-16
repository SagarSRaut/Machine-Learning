import matplotlib.pylab as plt
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import os


fp_df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\Cosmetics.csv", index_col=0)
fp_df.head()

fp_df.info()
# create a frequent itemsets
fp_df = fp_df.astype(bool)
itemsets = apriori(fp_df, min_support=0.2, use_colnames=True)


# and convert into rules

rules = association_rules(itemsets, metric='confidence', min_threshold=0.6)
rules.info()
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

rules.sort_values(by=['lift'], ascending=False).head(6)

rules_df = rules.sort_values(by=['lift', 'confidence'], ascending=False)


relv_rules = rules_df[rules_df['lift']>1]
relv_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]