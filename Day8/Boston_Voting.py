import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold
from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, VotingRegressor

boston = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\Boston.csv")
y = boston['medv']
X = boston.drop('medv', axis=1)

lr = LinearRegression()
ridge = Ridge()
lasso = Lasso()
dtr = DecisionTreeRegressor(random_state=24)

# min_samples_split is the minimum number of samples on which the tree will split

voting = VotingRegressor([('LR',lr),('RIDGE',ridge),('LASSO',lasso), 
                           ('DTR', dtr)])



X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   random_state=24, test_size=0.3)


voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)
print('Voting w/o weight',r2_voting)


lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
r2_lr = r2_score(y_test, y_pred)
print('LR',r2_lr)

lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
r2_lasso = r2_score(y_test, y_pred)
print('Lasso',r2_lasso)


ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_test)
r2_ridge = r2_score(y_test, y_pred)
print('Ridge',r2_ridge)

dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
r2_dtr = r2_score(y_test, y_pred)
print('DTR',r2_dtr)

############ Weighted Average

voting = VotingRegressor([('LR',lr),('RIDGE',ridge),('LASSO',lasso), 
                           ('DTR', dtr)], weights=[r2_lr,r2_lasso,r2_ridge,r2_dtr])

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
r2_voting = r2_score(y_test, y_pred)
print('Voting weight',r2_voting)




















############################################################################

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
print(voting.get_params())

params = {'SVML__SVM__C': np.linspace(0.001,3,5),
          'TREE__max_depth':[None,2,3],
          'SVMR__SVM__gamma':np.linspace(0.001,3,5),
          'SVMR__SVM__C': np.linspace(0.001,3,5),
          'LR__C': np.linspace(0.001,3,5)}

gcv = GridSearchCV(voting,param_grid=params,cv=kfold, 
                   scoring='neg_log_loss', n_jobs=-1)

##  n_jobs ==> assigns tasks paralelly to the processor
## -1 indicates: all the algorithms will be executed paralelly
##  0,1,2,3 will indicate: No algorithms, 1,2,3 algorithms will be executed paralelly

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)








