import pandas as pd
import os
os.chdir(r"C:\Training\Academy\Statistics (Python)\20. Recommender Systems\Amazon Movie Ratings")

df = pd.read_csv("Amazon.csv")

ratings = pd.melt(df, id_vars='user_id', 
                  var_name="item_id",value_name="rating")

ratings = ratings[ratings['rating'].notna()]


