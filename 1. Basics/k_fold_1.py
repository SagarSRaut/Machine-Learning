import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LogisticRegression 

sonar = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Sonar\Sonar.csv")
le = LabelEncoder()
y = le.fit_transform(sonar['Class'])
X = sonar.drop('Class', axis=1)
print(le.classes_)
gaussian = GaussianNB()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
### Scoring by accuracy
results = cross_val_score(gaussian, X, y,cv=kfold)
print(results.mean())
### Scoring by ROC AUC
results = cross_val_score(gaussian, X, y, cv=kfold, scoring='roc_auc')
print(results.mean())

lr = LogisticRegression()
results = cross_val_score(lr, X, y, cv=kfold, scoring='roc_auc')
print(results.mean())

