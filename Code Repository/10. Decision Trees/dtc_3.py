import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
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


params = {'min_samples_split':np.arange(2,35,5),
          'min_samples_leaf':np.arange(1, 35, 5),
          'max_depth':[None, 4, 3, 2]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(dtc, param_grid=params, verbose=3,
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X_train, y_train)
print(gcv.best_params_)
print(gcv.best_score_)

best_tree = gcv.best_estimator_
plt.figure(figsize=(50,20))
plot_tree(best_tree,feature_names=list(X_train.columns),
               class_names=list(le.classes_),
               filled=True,fontsize=25)
plt.title("Best Tree")
plt.show() 

df_imp = pd.DataFrame({'Feature':list(X_train.columns),
                       'Importance':best_tree.feature_importances_})

plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importances")
plt.show()

### Inferencing
dum_tst = pd.get_dummies(test, drop_first=True)
y_pred_prob = best_tree.predict_proba(dum_tst)

submit = pd.DataFrame({'id':list(test.index),
                       'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv("sbt_dtc.csv", index=False)

