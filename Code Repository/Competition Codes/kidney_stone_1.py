import pandas as pd 
import numpy as np
from sklearn.naive_bayes import GaussianNB
import os 

os.chdir(r"C:\Training\Kaggle\Competitions\Playground Competitions\Kidney Stone Prediction")

train = pd.read_csv("train.csv", index_col=0)
print(train.isnull().sum().sum())
test = pd.read_csv("test.csv")
print(test.isnull().sum().sum())

X_train = train.drop('target', axis=1)
y_train = train['target']
X_test = test.drop('id', axis=1)

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_prob = nb.predict_proba(X_test)[:,1]

submit = pd.DataFrame({'id':test['id'],
                       'target':y_pred_prob})
submit.to_csv("sbt_nb.csv", index=False)

