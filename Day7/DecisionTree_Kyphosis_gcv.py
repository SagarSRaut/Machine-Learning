import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge,ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

kyp = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Kyphosis\Kyphosis.csv')
kyp.head()


#Performs one-hot encoding using pandas
le = LabelEncoder()
y=le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis',axis=1)


# min_samples_split is the minimum number of samples on which the tree will split
dtc = DecisionTreeClassifier(random_state=24, min_samples_split=4)
dtc.fit(X, y)

plt.figure(figsize=(25,20))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=18)
plt.show() 

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   stratify=y,
                                   random_state=24)

dtc = DecisionTreeClassifier(random_state=24, max_depth=1)
dtc.fit(X_train, y_train)

plt.figure(figsize=(25,20))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=32)
plt.show() 

y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))


###############################################
# 1)  min_samples_split is the minimum number of samples on which the tree will split
# 2) min_samples_leaf is the minimum number of samples that the leaf node can have
# 3) 

params = {'min_samples_split':[2,4,6,10,20],  
          'min_samples_leaf': [1,5,10,15],
          'max_depth': [None,4,3,2]}


kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=24 )
gcv = GridSearchCV(dtc,param_grid=params,cv=kfold, scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


best_tree = gcv.best_estimator_
plt.figure(figsize=(25,20))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=18)
plt.show()