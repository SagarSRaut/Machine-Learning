import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score 

boston = pd.read_csv("Boston.csv")
y = boston['medv']
X = boston.drop('medv', axis=1)

knn = KNeighborsRegressor(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

y_pred_prob = knn.predict(X_test)
print(r2_score(y_test, y_pred))


################## Grid Search #################
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=24)
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10]}
knn = KNeighborsRegressor()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold,
                   scoring='r2')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
