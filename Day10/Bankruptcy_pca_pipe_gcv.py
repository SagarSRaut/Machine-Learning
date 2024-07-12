import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold
from sklearn.linear_model import LinearRegression, Ridge,ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, BaggingClassifier,StackingClassifier,GradientBoostingClassifier,RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

bankruptcy = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Bankruptcy\Bankruptcy.csv')
y = bankruptcy['D']
X = bankruptcy.drop(['NO','D'], axis=1)

lr = LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)

scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components = 11).set_output(transform='pandas')

pipe = Pipeline([('SCL', scaler),('PCA', prcomp), ('LR', lr)])
pipe.fit(X_train, y_train)
print(np.cumsum(prcomp.explained_variance_ratio_*100))

y_pred = pipe.predict(X_test)
y_pred_prob = pipe.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))




###########################grid search######################
print(pipe.get_params())
params= {'PCA__n_components': np.arange(6,12),
         'LR__C':np.linspace(0.001,3,5)}
kfold= StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)








