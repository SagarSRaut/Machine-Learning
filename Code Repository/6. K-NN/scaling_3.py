import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

boston = pd.read_csv("Boston.csv")
y = boston['medv']
X = boston.drop('medv', axis=1)

################## Grid Search #################
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=24)
knn = KNeighborsRegressor()
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
pipe = Pipeline([('SCL',None),('KNN',knn)])
params = {'KNN__n_neighbors':[1,2,3,4,5,6,7,8,9,10],
          'SCL':[None, std_scaler, mm_scaler]}

gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
