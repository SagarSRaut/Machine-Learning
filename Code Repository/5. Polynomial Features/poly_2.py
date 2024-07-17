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

boston = pd.read_csv("Boston.csv")
y = boston['medv']
X = boston.drop('medv', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24)

poly = PolynomialFeatures(degree=2).set_output(transform="pandas")
lr = LinearRegression()
pipe = Pipeline([('POLY',poly),('LR',lr)])
pipe.fit(X_train, y_train)
# poly.fit(X_train)
# X_poly_trn = poly.transform(X_train)
# lr.fit(X_poly_trn, y_train)

y_pred = pipe.predict(X_test)
# X_poly_tst = poly.transform(X_test)
# y_pred = lr.predict(X_poly_tst)

print(r2_score(y_test, y_pred))

################ k-FOLD #######################
kfold = KFold(n_splits=5, shuffle=True, random_state=24)
lr = LinearRegression()
degrees = [1,2,3,4,5]
scores = []
for i in degrees:
    poly = PolynomialFeatures(degree=i)
    pipe = Pipeline([('POLY',poly),('LR',lr)])
    results = cross_val_score(pipe, X, y, cv=kfold)
    scores.append(results.mean())

i_max =  np.argmax(scores)
print("Best degree =", degrees[i_max])
print("Best Score =", scores[i_max])

#### using grid search
print(pipe.get_params())
params = {'POLY__degree':[1,2,3,4,5]}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)

#### Ridge
ridge = Ridge()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',ridge)])
print(pipe.get_params())
params = {'POLY__degree':[1,2,3],
          'LR__alpha':np.linspace(0.001, 5, 10)}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)

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

best_model = gcv_lass.best_estimator_
print(best_model.named_steps.LR.coef_)
print(best_model.named_steps.LR.intercept_)

#### Elastic
elastic = ElasticNet()
poly = PolynomialFeatures()
pipe = Pipeline([('POLY',poly),('LR',elastic)])
print(pipe.get_params())
params = {'POLY__degree':[1,2,3],
          'LR__alpha':np.linspace(0.001, 5, 10),
          'LR__l1_ratio':np.linspace(0, 1, 5)}
gcv = GridSearchCV(pipe, param_grid=params,cv=kfold)
gcv.fit(X, y)
print(gcv.best_score_)
print(gcv.best_params_)
