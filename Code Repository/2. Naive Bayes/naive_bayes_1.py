import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


cancer = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Cancer\Cancer.csv")
# Hot Encoding
dum_can = pd.get_dummies(cancer, drop_first=True)
y = dum_can['Class_recurrence-events']
X = dum_can.drop(['Class_recurrence-events','subjid'],
                 axis=1)
# Instantiating the BernoulliNB class
nb = BernoulliNB()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
# Calculating posterior probabilities
y_pred_prob = nb.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]))
print(log_loss(y_test, y_pred_prob))

##### K-Fold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
### Scoring by accuracy
results = cross_val_score(nb, X, y,cv=kfold)
print(results.mean())
### Scoring by ROC AUC
results = cross_val_score(nb, X, y, cv=kfold, scoring='roc_auc')
print(results.mean())
### Scoring by log loss
results = cross_val_score(nb, X, y, 
                          cv=kfold, 
                          scoring='neg_log_loss')
print(results.mean())
