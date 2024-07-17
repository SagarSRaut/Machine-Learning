import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 

chem = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Chemical Process Data\ChemicalProcess.csv")
y = chem['Yield']
X = chem.drop('Yield', axis=1)

print(X.isnull().sum())
print(X.isnull().sum().sum())

# imp = SimpleImputer(strategy='mean').set_output(transform='pandas')
# X_imputed = imp.fit_transform(X)
# print(X_imputed.isnull().sum().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24)
imp = SimpleImputer(strategy='mean').set_output(transform='pandas')
# X_imp_trn = imp.fit_transform(X_train)
# X_imp_tst = imp.transform(X_test)

lr = LinearRegression()
pipe = Pipeline([('IMP',imp),('LR',lr)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(r2_score(y_test, y_pred))


############# Grid Search ##########################
#### LR
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=24)
imp = SimpleImputer()
lr = LinearRegression()
pipe = Pipeline([('IMP',imp),('LR',lr)])

params = {'IMP__strategy':['mean','median']}

gcv = GridSearchCV(pipe, param_grid=params, cv=kfold)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

### K-NN
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=24)
imp = SimpleImputer()
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
knn = KNeighborsRegressor()
pipe = Pipeline([('IMP',imp),('SCL',None),('KNN',knn)])

params = {'IMP__strategy':['mean','median'],
          'KNN__n_neighbors':[1,2,3,4,5,6,7,8,9,10],
          'SCL':[std_scaler, mm_scaler, None]}

gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   verbose=2)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

