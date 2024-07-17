import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
import numpy as np 

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
lr = LogisticRegression()
svm = SVC(kernel='linear', probability=True, random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
rf = RandomForestClassifier(random_state=24)
stack = StackingClassifier([('LR',lr),('SVM',svm),('TREE',dtc)], 
                           final_estimator=rf)

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = stack.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

### with pass through
stack = StackingClassifier([('LR',lr),('SVM',svm),('TREE',dtc)], 
                           final_estimator=rf,passthrough=True)

stack.fit(X_train, y_train)
y_pred = stack.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = stack.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#################################################
print( stack.get_params() )
params = {'SVM__C':np.linspace(0.01, 3, 5),
          'LR__C':np.linspace(0.01, 3, 5),
          'TREE__max_depth':[None, 2, 3, 4],
          'final_estimator__max_features':[2,3],
          'passthrough':[False, True]}
kfold = StratifiedKFold(n_splits=5, random_state=24, shuffle=True)
gcv = GridSearchCV(stack, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit( X, y )
print(gcv.best_score_)
print(gcv.best_params_)

