import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
 
conc = pd.read_csv("Concrete_Data.csv")
X = conc.drop('Strength', axis=1)
y = conc['Strength']

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
                   scoring='neg_mean_squared_error')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

### Unlabelled data
tst_conc = pd.read_csv("testConcrete.csv")
pred_strength = best_model.predict(tst_conc)



