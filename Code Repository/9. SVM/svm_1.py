import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np 

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

svm = SVC(C=0.5, kernel="linear", 
          probability=True, random_state=24)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = svm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

##########################################
params = {'C':[0.1, 1, 0.5, 2, 3]}
svm = SVC(kernel="linear", probability=True, random_state=24)
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv = GridSearchCV(svm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

#######################################################
## with scaling
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
svm = SVC(kernel="linear", probability=True, random_state=24)
pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C':np.linspace(0.001, 5, 20) ,
          'SCL':[None, std_scaler, mm_scaler]}

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

#### Poly
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
svm = SVC(kernel="poly", probability=True, random_state=24)
pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C':np.linspace(0.001, 5, 20) ,
          'SCL':[None, std_scaler, mm_scaler],
          'SVM__degree':[2,3],
          'SVM__coef0':np.linspace(0, 3, 5)}

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss', verbose=2)
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

#### Radial
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
svm = SVC(kernel="rbf", probability=True, random_state=24)
pipe = Pipeline([('SCL',None),('SVM',svm)])
params = {'SVM__C':np.linspace(0.001, 5, 20) ,
          'SCL':[None, std_scaler, mm_scaler],
          'SVM__gamma':np.linspace(0.001, 5, 5) }

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss', verbose=2)
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)
