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
from sklearn.ensemble import VotingRegressor, BaggingRegressor
import numpy as np 
import os 

os.chdir(r"C:\Training\Kaggle\Competitions\Playground Competitions\Flood Prediction")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('FloodProbability', axis=1)
y_train = train['FloodProbability']
X_test = test.drop('id', axis=1)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability':y_pred})
submit.to_csv("sbt_lr.csv", index=False)

################# Randomized Search CV ###########

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

voting = VotingRegressor([('LR',lr),('RIDGE',ridge),
                           ('LASSO', lasso),('TREE', dtr)])
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
                   scoring='r2', n_jobs=-1, n_iter=10)
rgcv.fit(X_train, y_train)
pd_rgcv = pd.DataFrame( rgcv.cv_results_ ) 
print(rgcv.best_params_)
print(rgcv.best_score_)

best_model = rgcv.best_estimator_ 

y_pred = best_model.predict(X_test)
submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability':y_pred})
submit.to_csv("sbt_vote_rgcv_10.csv", index=False)

#######

