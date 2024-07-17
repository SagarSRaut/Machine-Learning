import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree


df = pd.read_csv("Housing.csv")
ohc = OneHotEncoder(sparse_output=False, drop='first')
ct = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_include=['int64',
        'float64'])), 
       verbose_feature_names_out=False ).set_output(transform='pandas')
dum_pd = ct.fit_transform(df)

y = dum_pd['price']
X = dum_pd.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24)

dtr = DecisionTreeRegressor(random_state=24, max_depth=2)
dtr.fit(X_train, y_train)

plt.figure(figsize=(35,20))
plot_tree(dtr,feature_names=list(X.columns),
               filled=True,fontsize=28)
plt.show() 

y_pred = dtr.predict(X_test)
print(r2_score(y_test, y_pred))

##################################################

params = {'min_samples_split':np.arange(2,35,5),
          'min_samples_leaf':np.arange(1, 35, 5),
          'max_depth':[None, 4, 3, 2]}
kfold = KFold(n_splits=5, shuffle=True,
                        random_state=24)
dtc = DecisionTreeRegressor(random_state=24)
gcv = GridSearchCV(dtr, param_grid=params,verbose=3, 
                   cv=kfold, scoring="r2")
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)
