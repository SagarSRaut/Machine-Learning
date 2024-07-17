import pandas as pd
from sklearn.model_selection import train_test_split 
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor 
from lightgbm import LGBMRegressor
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

c_gbm = CatBoostRegressor(random_state=24)

c_gbm.fit(X_train, y_train)
y_pred = c_gbm.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability':y_pred})
submit.to_csv("sbt_c_gbm.csv", index=False)

##### X G Boost
x_gbm = XGBRegressor(random_state=24)

x_gbm.fit(X_train, y_train)
y_pred = x_gbm.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability':y_pred})
submit.to_csv("sbt_x_gbm.csv", index=False)


##### Light GBM
l_gbm = LGBMRegressor(random_state=24)

l_gbm.fit(X_train, y_train)
y_pred = l_gbm.predict(X_test)

submit = pd.DataFrame({'id':test['id'],
                       'FloodProbability':y_pred})
submit.to_csv("sbt_l_gbm.csv", index=False)
