import gradio as gr 
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import httpx
from mlxtend.preprocessing import TransactionEncoder




groceries=[]
with open(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\Groceries.csv","r")as f:groceries=f.read()
groceries=groceries.split("\n")

groceries_list=[]
for i in groceries:
    groceries_list.append(i.split(","))
te= TransactionEncoder()
te_ary=te.fit(groceries_list).transform(groceries_list)
te_ary

fp_df=pd.DataFrame(te_ary,columns=te.columns_)

fp_df = fp_df.astype(bool)

def gen_rules(min_sup, min_conf):
    itemsets = apriori(fp_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(itemsets, metric='confidence', min_threshold=min_conf)
    rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    rules = rules[rules['lift']>1]
    rules['antecedents'] = [list(x) for x in rules['antecedents'].values]
    rules['consequents'] = [list(x) for x in rules['consequents'].values]
    return rules.sort_values(by='lift', ascending=False)


demo = gr.Interface(gen_rules, 
                    inputs= [gr.Slider(value=0.01, step=0.01,
                                   label="Minimum Support",
                                   minimum=0.0001, maximum=1),
                             gr.Slider(value=0.01, step=0.01,
                                   label="Minimum Confidence",    
                                   minimum=0.0001, maximum=1)], 
                    outputs='dataframe')

if __name__ == "__main__":
    demo.launch()
    
    
    
