from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
import numpy as np 

a = pd.DataFrame({'x1':[10, 9, 11],
                  'x2':[0.1, 0.7, 0.1]})

scl_std = StandardScaler()

scl_std.fit(a)
## means
print(scl_std.mean_)
## sd
print(scl_std.scale_)

scl_std.transform(a)

#or
scl_std.fit_transform(a)

################################################
scl_mm = MinMaxScaler()

scl_mm.fit(a)
print(scl_mm.data_min_)
print(scl_mm.data_max_)

scl_mm.transform(a)

