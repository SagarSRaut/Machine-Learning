import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np 

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

gbm = GradientBoostingClassifier(random_state=24)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = gbm.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gbm = GradientBoostingClassifier(random_state=24)
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
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
gbm = GradientBoostingClassifier(random_state=24)
gcv = GridSearchCV(gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)
