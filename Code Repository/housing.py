#########################  Q1  #######################
s ={}

!pip install xgboost

import pandas as pd
import seaborn as sns

housing = pd.read_csv('Datasets/Housing.csv')

housing

# XGBRegressor

from xgboost import XGBRegressor
from sklearn.model_selection import KFold,GridSearchCV,train_test_split

X = housing.drop('price',axis = 1)
X = pd.get_dummies(X,drop_first=True)
y = housing[['price']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=24)

xgb = XGBRegressor()
xgb.get_params()

params = {
    'learning_rate':[0.001,0.3],
    'max_depth':[2,5],
    'n_estimators':[10,20]
}

kfold = KFold(n_splits=5,shuffle=True,random_state=24)

gcv = GridSearchCV(xgb,param_grid=params,cv=kfold,scoring='neg_mean_squared_error', verbose=1)
gcv1 = GridSearchCV(xgb,param_grid=params,cv=kfold,scoring='r2', verbose=1)

gcv.fit(X,y)
gcv1.fit(X, y)

pd_cv = pd.DataFrame(gcv.cv_results_)
xgb_r2 = gcv1.best_score_
s['xgb_r2'] = xgb_r2
print(gcv.best_params_)
print(gcv.best_score_)
print(gcv1.best_params_)
print(gcv1.best_score_)

# # Fit the Grid Search model on the training data
# gcv.fit(X_train, y_train)

# # Get the best estimator and its parameters
# best_estimator = gcv.best_estimator_
# best_params = gcv.best_params_

# # Print the best parameters
# print("Best Parameters:", best_params)

# # Evaluate the model's performance on the test data
# score = gcv.score(X_test, y_test)
# print("R-squared Score on Test Data:", score)

# Decision Tree Regressor

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt.get_params()

params = {'max_depth': [3, None],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 5]
         }

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

gcv = GridSearchCV(dt, param_grid=params, cv=kfold, scoring='r2')

gcv.fit(X, y)

pd_gv = pd.DataFrame(gcv.cv_results_)

print(gcv.best_params_)
print(gcv.best_score_)
dt_r2 = gcv.best_score_
s['dt_r2'] =  dt_r2

# Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.get_params()

params = {'max_features' : [3,4,5]}

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

gcv = GridSearchCV(rf,param_grid=params,cv=kfold,scoring='r2')

gcv.fit(X, y)

pd_gv = pd.DataFrame(gcv.cv_results_)
pd_gv

gcv.best_params_

gcv.best_score_

rf_r2 = gcv.best_score_
s['rf_r2'] = rf_r2

s

m=max(s.values())
for i in s:
    if s[i] == m:
        print(f'{i}={m}')

###########################   Q2   ###############################
# KNN

import pandas as pd
glass = pd.read_csv('Cases/Glass Identification/glass.csv')

glass

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X = glass.drop('Type',axis=1)
y = le.fit_transform(glass[['Type']])

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.get_params()

params = {
    'n_neighbors':[1,3,2,4,5],
    'metric':['manhattan']
}

from sklearn.model_selection import StratifiedKFold,cross_val_score

kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=24)

result = cross_val_score(knn, X,y, cv=kfold)

result.mean()

from sklearn.model_selection import GridSearchCV

gcv = GridSearchCV(knn,param_grid=params,cv = kfold)
gcv.fit(X,y)

gcv.best_params_

gcv.best_score_

K_score= gcv.score(X, y)

gcv.cv_results_

# Isolation Forest

glass['Type'] = le.fit_transform(glass['Type'])


from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import seaborn as sns 
import numpy as np

############################################################################

clf = IsolationForest(contamination=0.01, random_state=24)
clf.fit(glass)
predictions = clf.predict(glass)

print("percentage of outliers="+ str((predictions<0).mean()*100)+ "%")
abn_ind = np.where(predictions < 0)
print("Outliers:")
print(glass.index[abn_ind])



###### Visualization of Outliers ################

scaler = StandardScaler()
scaled_df = scaler.fit_transform(glass)
prcomp = PCA()
scores = prcomp.fit_transform(scaled_df)
print(scores)
print(np.cumsum(prcomp.explained_variance_ratio_))

obs = np.where(predictions == -1,  "Outlier", "Inlier")
PCs = pd.DataFrame({'PC1':scores[:,0], 'PC2':scores[:,1],
                    'Class':obs})

sns.scatterplot(data=PCs, x='PC1',
                y='PC2', hue='Class')
for i in np.arange(0, glass.shape[0] ):
    plt.text(scores[i,0], scores[i,1], 
             list(glass.index)[i],fontsize=6)
plt.legend(loc='best')
plt.title("PCA")
plt.show()


# Support Vector 

from sklearn.svm import SVC

le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis=1)
print(le.classes_)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
svm = SVC(kernel="linear", probability=True, random_state=24)

pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C':np.linspace(0.001, 5, 20) ,
          'SCL':[None, std_scaler, mm_scaler],
          'SVM__decision_function_shape':['ovr']}

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv_lin = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')

gcv_lin.fit(X, y)
pd_cv = pd.DataFrame( gcv_lin.cv_results_ )
print(gcv_lin.best_params_)
print(gcv_lin.best_score_)

s_score = gcv.score(X,y)

print('Score of Kneighbour ', K_score ,'\nScore of SVM ', s_score)

# Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.get_params()

params = {'max_features' : [3]}

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

gcv = GridSearchCV(rf,param_grid=params,cv=kfold)

gcv.fit(X, y)

pd_gv = pd.DataFrame(gcv.cv_results_)
pd_gv

gcv.best_params_

gcv.best_score_

rf_r2 = gcv.best_score_

print('Score of Kneighbour ', K_score ,'\nScore of SVM ', s_score,'\nScore of Random Forest',rf_r2)



