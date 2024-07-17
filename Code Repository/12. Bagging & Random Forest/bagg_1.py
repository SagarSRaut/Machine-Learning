import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split 

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

lr = LogisticRegression()
dtc = DecisionTreeClassifier(random_state=24)
bagg = BaggingClassifier(dtc, n_estimators=25,random_state=24,
                         n_jobs=-1, oob_score=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify=y)
bagg.fit(X_train, y_train)
print("OOB score:", bagg.oob_score_)

y_pred = bagg.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = bagg.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

######## Grid Search CV #########
print(bagg.get_params())
params = {'estimator':[lr, dtc]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
gcv = GridSearchCV(bagg, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)