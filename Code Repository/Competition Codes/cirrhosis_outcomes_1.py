import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree 
import os 
os.chdir(r"C:\Training\Kaggle\Competitions\Playground Competitions\Cirrhosis Outcomes")

train = pd.read_csv("train.csv", index_col=0)
test = pd.read_csv("test.csv", index_col=0)

X_train = pd.get_dummies(train.drop('Status', axis=1), 
                         drop_first=True)
le = LabelEncoder()
y_train = le.fit_transform(train['Status'])
print(le.classes_)

dtc = DecisionTreeClassifier(random_state=24)
bagg = BaggingClassifier(dtc, random_state=24,
                         n_estimators=30)
print(bagg.get_params())
params = {'estimator__min_samples_split':np.arange(2,35,5),
          'estimator__min_samples_leaf':np.arange(1, 35, 5),
          'estimator__max_depth':[None, 4, 3, 2]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)

gcv = GridSearchCV(bagg, param_grid=params, verbose=3,
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X_train, y_train)
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
submit.to_csv("sbt_bagg.csv", index=False)

