from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV,StratifiedKFold
from sklearn.metrics import r2_score ,accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer, make_column_selector

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Competition_Kaggle\playground-series-s3e26 (1)\train.csv"
                    ,index_col=0)
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Competition_Kaggle\playground-series-s3e26 (1)\test.csv")

X_train = train.drop('Status',axis=1)  #take all the columns upto the salary column
y_train = train['Status']
X_test = test.drop('id',axis=1)


lgbm=LGBMClassifier(random_state=24)


lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
y_pred_prob=lgbm.predict_proba(X_test)
# print(log_loss(y_test, y_pred_prob))



submit = pd.DataFrame({'id':test['id'],'Status_C':y_pred_prob[:,0],
                       'Status_CL':y_pred_prob[:,1],
                       'Status_D':y_pred_prob[:,2]})
submit.to_csv('sbt_cirrhosis_catboost.csv',index=False)
















