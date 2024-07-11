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



kyp = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Kyphosis\Kyphosis.csv')
kyp.head()

#Performs one-hot encoding using pandas
le = LabelEncoder()
y=le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                   random_state=24, test_size=0.3)

lr=LogisticRegression()
svm=SVC(kernel='linear',probability=True,random_state=24)
dtc=DecisionTreeClassifier(random_state=24)
rf=RandomForestClassifier(random_state=24)
stack=StackingClassifier([('LR',lr),('SVM',svm),('TREE',dtc)],final_estimator=rf)
 
stack.fit(X_train,y_train)
y_pred=stack.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_pred_proba=stack.predict_proba(X_test)
print(log_loss(y_test, y_pred_proba))      


###############################################
print(stack.get_params())
params = {'SVM__C':np.arange(0.01,3,5),  
          'LR__C': np.arange(0.01,1,5),
          'TREE__max_depth': [None,2,3,4],
          'final_estimator__max_features':[2,3],
          'passthrough':[False,True]}


kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=24 )
gcv = GridSearchCV(stack,param_grid=params,cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
y_pred= gcv.predict(X_test)

y_pred_prob = gcv.predict_proba(X_test)
print(gcv.best_params_)
print(gcv.best_score_)
