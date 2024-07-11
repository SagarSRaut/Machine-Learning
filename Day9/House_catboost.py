import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV, StratifiedKFold, cross_val_score,KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, r2_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler
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
from catboost import CatBoostClassifier,CatBoostRegressor

housing = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\Housing.csv")
 
X = housing.drop('price', axis=1)
y = housing['price']

housing.info()
housing

cat = list(X.select_dtypes(include=object).columns)

cb = CatBoostRegressor(random_state=24, cat_features=cat)


X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   random_state=24, test_size=0.3)


cb.fit(X_train,y_train)

y_pred = cb.predict(X_test)
print(r2_score(y_test,y_pred))

################### CatBoost ########################

print(cb.get_params())
kfold = KFold(n_splits=5, shuffle=True, random_state=24)

params = {'learning_rate':np.linspace(0.001,0.9,10),
          'max_depth':[None,2,3,4],
          'n_estimators':[25,50,100]}

gcv=GridSearchCV(cb, param_grid=params,cv=kfold,
                        n_jobs=-1)
gcv.fit(X_train,y_train)
pd_cv = pd.DataFrame(gcv.cv_results_)
print(gcv.best_params_)
print('R2 score: ',gcv.best_score_)