import pandas as pd
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np 

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

x_gbm = XGBClassifier(random_state=24)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
x_gbm.fit(X_train, y_train)
y_pred = x_gbm.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = x_gbm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
x_gbm = XGBClassifier(random_state=24)
gcv = GridSearchCV(x_gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

##### Sonar
sonar = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Sonar\Sonar.csv")
le = LabelEncoder()
y = le.fit_transform(sonar['Class'])
X = sonar.drop('Class', axis=1)
print(le.classes_)

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
x_gbm = XGBClassifier(random_state=24)
gcv = GridSearchCV(x_gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

################# Light GBM ##################
from lightgbm import LGBMClassifier


kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

l_gbm = LGBMClassifier(random_state=24)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
l_gbm.fit(X_train, y_train)
y_pred = l_gbm.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = l_gbm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
l_gbm = LGBMClassifier(random_state=24)
gcv = GridSearchCV(l_gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)



##### Sonar
sonar = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Sonar\Sonar.csv")
le = LabelEncoder()
y = le.fit_transform(sonar['Class'])
X = sonar.drop('Class', axis=1)
print(le.classes_)

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
x_gbm = XGBClassifier(random_state=24)
gcv = GridSearchCV(x_gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)



############ Cat Boost #################
from catboost import CatBoostClassifier
kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

c_gbm = CatBoostClassifier(random_state=24)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
c_gbm.fit(X_train, y_train)
y_pred = c_gbm.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = c_gbm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
c_gbm = CatBoostClassifier(random_state=24)
gcv = GridSearchCV(c_gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)



##### Sonar
sonar = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Sonar\Sonar.csv")
le = LabelEncoder()
y = le.fit_transform(sonar['Class'])
X = sonar.drop('Class', axis=1)
print(le.classes_)

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
x_gbm = XGBClassifier(random_state=24)
gcv = GridSearchCV(x_gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)



