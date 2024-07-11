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
from sklearn.ensemble import VotingClassifier, BaggingClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")


sonar = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Sonar\Sonar.csv")
le=LabelEncoder()


y=le.fit_transform(sonar['Class']) 
X= sonar.drop('Class',axis=1)
print(le.classes_)

# lr = LogisticRegression()
# dtr = DecisionTreeRegressor(random_state=24)

lgbm = LGBMClassifier(random_state=24)
x_gbm = XGBClassifier(random_state=24)
gbm= GradientBoostingClassifier(random_state=24)
cb = CatBoostClassifier(random_state=24)
# bagg = BaggingClassifier(dtr, n_estimators=25, n_jobs=-1,random_state=24)


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                   random_state=24, test_size=0.3)


cb.fit(X_train,y_train)

y_pred = cb.predict(X_test)
print(accuracy_score(y_test,y_pred))
y_pred_prob = cb.predict_proba(X_test)
print(log_loss(y_test,y_pred_prob))

################### LGBM XGB GB CatBoost########################

print(cb.get_params())
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)

params = {'learning_rate':np.linspace(0.001,0.9,10),
          'max_depth':[None,2,3,4],
          'n_estimators':[25,50,100]}

gcv=GridSearchCV(cb, param_grid=params,cv=kfold,
                        scoring='neg_log_loss', n_jobs=-1)
gcv.fit(X_train,y_train)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print(gcv.best_score_)


# Simple random sampling without replacement (SRSWOR)
# kyp_ind=list(kyp.index)
# #replace=False indicates, sampling happens without replacement
# samp_ind= np.random.choice(kyp_ind,size=60,replace=False)

# samp_kyp = kyp.iloc[samp_ind,:]


