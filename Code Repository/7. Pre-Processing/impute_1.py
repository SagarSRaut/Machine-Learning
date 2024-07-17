import numpy as np 
import pandas as pd
from sklearn.impute import SimpleImputer 

a = np.array([[23, np.nan, 45],
              [24, 56 , 67],
              [np.nan, 59, 68],
              [26, 52, np.nan],
              [26, 50, 60],
              [np.nan, 56, 70]])

pa = pd.DataFrame(a, columns=['x1','x2','x3'])

## Counting no. of missings in different columns
print(pa.isnull().sum())

### Constant Imputation

imp = SimpleImputer(strategy='constant',
      fill_value=30)
imp_pa = imp.fit_transform(pa)
type(imp_pa)

imp = SimpleImputer(strategy='constant',
      fill_value=30).set_output(transform="pandas")
imp_pa = imp.fit_transform(pa)
type(imp_pa)



### Mean Imputation
imp = SimpleImputer(strategy='mean')
imp.fit(pa)
print(imp.statistics_)
imp.transform(pa)

imp_pa = imp.fit_transform(pa)


### Median Imputation
imp = SimpleImputer(strategy='median')
imp.fit(pa)
print(imp.statistics_)
imp.transform(pa)

imp_pa = imp.fit_transform(pa)
