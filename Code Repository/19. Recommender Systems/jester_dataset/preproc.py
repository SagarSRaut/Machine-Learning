import pandas as pd
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\20. Recommender Systems\jester_dataset")

df = pd.read_excel("[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx",
                   header=None)

ratings = pd.melt(df, id_vars=0)
ratings.columns=["uid", "iid", "rating"]
ratings = ratings[(ratings['rating']<=10) & (ratings['rating']>=-10)]

