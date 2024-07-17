import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score 
from sklearn.pipeline import Pipeline 

conc = pd.read_csv("Concrete_Data.csv")
X = conc.drop('Strength', axis=1)
y = conc['Strength']

kfold = KFold(n_splits=5, shuffle=True, random_state=24)
#### Ridge
ridge = Ridge()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',ridge)])
print(pipe.get_params())
params = {'POLY__degree':[1,2,3],
          'LR__alpha':np.linspace(0.001, 5, 10)}
gcv_ridge = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv_ridge.fit(X, y)
print(gcv_ridge.best_score_)
print(gcv_ridge.best_params_)

#### Lasso
lasso = Lasso()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',lasso)])
print(pipe.get_params())
params = {'POLY__degree':[1,2,3],
          'LR__alpha':np.linspace(0.001, 5, 10)}
gcv_lass = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv_lass.fit(X, y)
print(gcv_lass.best_score_)
print(gcv_lass.best_params_)

#### Elastic
elastic = ElasticNet()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',elastic)])
print(pipe.get_params())
params = {'POLY__degree':[1,2,3],
          'LR__alpha':np.linspace(0.001, 5, 10),
          'LR__l1_ratio':np.linspace(0, 1, 5)}
gcv_el = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv_el.fit(X, y)
print(gcv_el.best_score_)
print(gcv_el.best_params_)

############ Inferencing ###############
###### in case refit = false
# poly = PolynomialFeatures(degree=3)
# ridge = Ridge(alpha=5)
# best_model = Pipeline([('POLY',poly),('LR',elastic)])
# best_model.fit(X, y)

best_model = gcv_ridge.best_estimator_

### Unlabelled data
tst_conc = pd.read_csv("testConcrete.csv")
pred_strength = best_model.predict(tst_conc)



