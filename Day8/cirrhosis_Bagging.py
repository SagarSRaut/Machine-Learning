import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold
from sklearn.linear_model import LinearRegression, Ridge,ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, BaggingClassifier
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
# import os 
# os.chdir(r"C:\Training\Kaggle\Competitions\Playground Competitions\Cirrhosis Outcomes")

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Competition_Kaggle\playground-series-s3e26 (1)\train.csv", index_col=0)
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Competition_Kaggle\playground-series-s3e26 (1)\test.csv", index_col=0)

X_train = pd.get_dummies(train.drop('Status', axis=1), 
                    drop_first=True)
le      = LabelEncoder()
y_train = le.fit_transform(train['Status'])
print(le.classes_)

X_test = pd.get_dummies(test, drop_first=True)

lr = LogisticRegression()
dtc = DecisionTreeClassifier()
bagg = BaggingClassifier(dtc,random_state=24)

kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)

bagg.fit(X_train,y_train)

params = {'estimator__min_samples_leaf':np.arange(2,35,5),
          'estimator__min_samples_split': np.arange(1,35,5),
          'estimator__max_depth': [None,4,3,2]}
gcv = GridSearchCV(bagg,param_grid=params,
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X_train, y_train)
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_
y_pred = best_model.predict(X_test)







# best_tree = gcv.best_estimator_
# plt.figure(figsize=(50,20))
# plot_tree(best_tree,feature_names=list(X_train.columns),
#                class_names=list(le.classes_),
#                filled=True,fontsize=25)
# plt.title("Best Tree")
# plt.show() 

# df_imp = pd.DataFrame({'Feature':list(X_train.columns),
#                        'Importance':best_tree.feature_importances_})

# plt.barh(df_imp['Feature'],
#         df_imp['Importance'])
# plt.title("Feature Importances")
# plt.show()

# ### Inferencing
# dum_tst = pd.get_dummies(test, drop_first=True)
# y_pred_prob = best_tree.predict_proba(dum_tst)

# submit = pd.DataFrame({'id':list(test.index),
#                        'Status_C':y_pred_prob[:,0],
#                        'Status_CL':y_pred_prob[:,1],
#                        'Status_D':y_pred_prob[:,2]})
# submit.to_csv("sbt_dtc.csv", index=False)

