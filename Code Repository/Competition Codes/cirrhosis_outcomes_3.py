import pandas as pd
from sklearn.model_selection import train_test_split 
from catboost import CatBoostClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np 
import os 

os.chdir(r"C:\Training\Kaggle\Competitions\Playground Competitions\Cirrhosis Outcomes")

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

X_train = pd.get_dummies(train.drop('Status', axis=1), 
                         drop_first=True)
le = LabelEncoder()
y_train = le.fit_transform(train['Status'])
print(le.classes_)

c_gbm = CatBoostClassifier(random_state=24)

c_gbm.fit(X_train, y_train)

### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = c_gbm.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_c_gbm.csv", index=False)

### XGBoost
x_gbm = XGBClassifier(random_state=24)

x_gbm.fit(X_train, y_train)

### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = x_gbm.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_x_gbm.csv", index=False)

### Light GBM
l_gbm = LGBMClassifier(random_state=24)

l_gbm.fit(X_train, y_train)

### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = l_gbm.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_l_gbm.csv", index=False)


#### Tuning Light GBM

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
l_gbm = LGBMClassifier(random_state=24)
gcv = GridSearchCV(l_gbm, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X_train, y_train)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = best_model.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_l_gbm_tuned.csv", index=False)