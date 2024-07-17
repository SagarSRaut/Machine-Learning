import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
import numpy as np 

boston = pd.read_csv("Boston.csv")
X = boston.drop('medv', axis=1)
y = boston['medv']

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

voting = VotingRegressor([('LR',lr),('RIDGE',ridge),
                           ('LASSO', lasso),('TREE', dtr)])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)

dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)

print("LR:", r2_lr)
print("Ridge:", r2_ridge)
print("Lasso:", r2_lasso)
print("Tree:", r2_dtr)
print("Voting:", r2_voting)


#### Weighted average
voting = VotingRegressor([('LR',lr),('RIDGE',ridge),
                           ('LASSO', lasso),('TREE', dtr)],
                         weights=[r2_lr,r2_ridge,
                                  r2_lasso,r2_dtr])
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)
print("Weighted Voting:", r2_voting)

#####################################################################
conc = pd.read_csv("Concrete_Data.csv")
X = conc.drop('Strength', axis=1)
y = conc['Strength']


voting = VotingRegressor([('LR',lr),('RIDGE',ridge),
                           ('LASSO', lasso),('TREE', dtr)])

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24)

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)

ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)

dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)

print("LR:", r2_lr)
print("Ridge:", r2_ridge)
print("Lasso:", r2_lasso)
print("Tree:", r2_dtr)
print("Voting:", r2_voting)


#### Weighted average
voting = VotingRegressor([('LR',lr),('RIDGE',ridge),
                           ('LASSO', lasso),('TREE', dtr)],
                         weights=[r2_lr,r2_ridge,
                                  r2_lasso,r2_dtr])
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)
print("Weighted Voting:", r2_voting)

################# Grid Search CV #########################
kfold = KFold(n_splits=5, shuffle=True, random_state=24)

voting = VotingRegressor([('LR',lr),('RIDGE',ridge),
                          ('LASSO', lasso),('TREE', dtr)])

print(voting.get_params())
params = {'RIDGE__alpha': np.linspace(0.001, 3, 5),
          'LASSO__alpha': np.linspace(0.001, 3, 5),
          'TREE__max_depth':[None, 3, 4, 5],
          'TREE__min_samples_split':[2, 5, 10],
          'TREE__min_samples_leaf': [1, 5, 10]}

gcv = GridSearchCV(voting, param_grid=params, cv=kfold,
                   scoring='r2', n_jobs=-1)
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################# Randomized Search CV ###########

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

voting = VotingRegressor([('LR',lr),('RIDGE',ridge),
                          ('LASSO', lasso),('TREE', dtr)])

print(voting.get_params())
params = {'RIDGE__alpha': np.linspace(0.001, 3, 10),
          'LASSO__alpha': np.linspace(0.001, 3, 10),
          'TREE__max_depth':[None, 3, 4, 5],
          'TREE__min_samples_split':[2,4,5,8,10],
          'TREE__min_samples_leaf': [1,4,5,8,10]}

rgcv = RandomizedSearchCV(voting, param_distributions=params,
                   cv=kfold, random_state=24,
                   scoring='r2', n_jobs=-1, n_iter=20)
rgcv.fit(X, y)
pd_rgcv = pd.DataFrame( rgcv.cv_results_ ) 
print(rgcv.best_params_)
print(rgcv.best_score_)
