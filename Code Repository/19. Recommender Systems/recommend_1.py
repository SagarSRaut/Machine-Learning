import numpy as np 
import pandas as pd 

a = [3,5,6,8,1,2,4] # all the items
b = [8,1,4] # items rated by a user

print(np.setdiff1d(a, b))
# items not rated by that user

### Reshaping the data
quality = pd.read_csv("C:/Training/Academy/Statistics (Python)/Datasets/quality.csv")
qual_melt = pd.melt(quality, id_vars='Sno')

qual_melt = pd.melt(quality, id_vars='Sno',
                    var_name="Category",value_name="Score")

qual_pivot = pd.pivot_table(qual_melt, index='Sno',
                            columns='Category',values='Score')
