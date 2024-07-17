import pandas as pd
from sklearn.model_selection import train_test_split 
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np 
import matplotlib.pyplot as plt 

housing = pd.read_csv("Housing.csv")

X = housing.drop('price', axis=1)
y = housing['price']

cat = list(X.select_dtypes(include=object).columns)

c_gbm = CatBoostRegressor(random_state=24,
                          cat_features=cat)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24)
c_gbm.fit(X_train, y_train)
y_pred = c_gbm.predict(X_test)
print(r2_score(y_test, y_pred))

params = {'learning_rate': np.linspace(0.001, 0.9, 10),
          'max_depth': [None, 2, 3, 4],
          'n_estimators': [25, 50, 100]}
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=24)
c_gbm = CatBoostRegressor(random_state=24,
                          cat_features=cat)
gcv = GridSearchCV(c_gbm, param_grid=params, cv=kfold)
gcv.fit(X, y)
pd_cv = pd.DataFrame( gcv.cv_results_ )
print(gcv.best_params_)
print(gcv.best_score_)

best_model = gcv.best_estimator_

df_imp = pd.DataFrame({'Feature':list(X.columns),
         'Importance':best_model.feature_importances_})

plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importances")
plt.show()
