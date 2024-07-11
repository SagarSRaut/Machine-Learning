from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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

house = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\Housing.csv")

X=house.drop('price', axis=1)
y=house['price']

#cat=['driveway','recroom','fullbase','gashw','prefarea','airco']
cat=list(X.select_dtypes(include=object).columns)
#svm=SVC(C=0.1,kernel='linear')

#gbm=GradientBoostingClassifier()
#lgbm=LGBMClassifier(random_state=24)
c_gbm=CatBoostRegressor(random_state=24,cat_features=cat)
#x_gbm=XGBClassifier(random_state=24)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                   random_state=24)
c_gbm.fit(X_train, y_train)
print('Train Test - Accuracy score and log loss')
y_pred = c_gbm.predict(X_test)
print(r2_score(y_test, y_pred))
# y_pred_prob=c_gbm.predict_proba(X_test)
# print(log_loss(y_test, y_pred_prob))

params={'learning_rate':np.linspace(0.001,0.9,10),
        'max_depth':[None,2,3,4],
        'n_estimators':[25,50,100]}

kfold=KFold(n_splits=5, shuffle= True,random_state=24)
gcv=GridSearchCV(c_gbm, param_grid=params,cv=kfold)
print('R2 Score')
gcv.fit(X,y)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(gcv.best_score_)
print(gcv.best_params_)
