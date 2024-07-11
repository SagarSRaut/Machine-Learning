from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso,ElasticNet
from sklearn.preprocessing import PolynomialFeatures,StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV,RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import r2_score ,accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer, make_column_selector

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Competition_Kaggle\playground-series-s4e5\train.csv",index_col=0)
print(train.isnull().sum())
test = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Competition_Kaggle\playground-series-s4e5\test.csv")
print(test.isnull().sum())

X_train = train.drop('FloodProbability',axis=1)  #take all the columns upto the salary column
y_train = train['FloodProbability']
X_test = test.drop('id',axis=1)

#cat=['driveway','recroom','fullbase','gashw','prefarea','airco']
#svm=SVC(C=0.1,kernel='linear')

#gbm=GradientBoostingClassifier()
#lgbm=LGBMClassifier(random_state=24)
c_gbm=CatBoostRegressor(random_state=24)
#x_gbm=XGBClassifier(random_state=24)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, 
#                                                     test_size = 0.3, 
#                                                    random_state=24)

kfold=KFold(n_splits=5, shuffle= True,random_state=24)

print('Train Test - Accuracy score and log loss')
y_pred = c_gbm.predict(X_test)
# print(r2_score(y_test, y_pred))
# y_pred_prob=c_gbm.predict_proba(X_test)
# print(log_loss(y_test, y_pred_prob))

params={'learning_rate':np.linspace(0.001,0.9,10),
        'max_depth':[None,2,3,4],
        'n_estimators':[25,50,100]}

rgcv=RandomizedSearchCV(voting, param_distributions=params,cv=kfold,random_state=24,
                        scoring='r2',n_iter=20)
rgcv.fit(X_train,y_train)
pd_cv=pd.DataFrame(gcv.cv_results_)
print(rgcv.best_params_)
print(rgcv.best_score_)

best_model = gcv.best_estimator_
y_pred = best_model.predict(X_test)

submit = pd.DataFrame({'id':test['id'],'FloodProbablity':y_pred})
submit.to_csv('sbt_flood_voting.csv',index=False)












