import pandas as pd
import numpy as np
import os
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import make_column_selector
from sklearn.linear_model import LinearRegression 
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV

os.chdir(r"C:\Training\AV\Big Mart III")

train = pd.read_csv("train_v9rqX0R.csv")
test = pd.read_csv("test_AbJTz2l.csv")
print(train.columns)
print(train.info())

# Item_Identifier
train['Item_Identifier'].value_counts()

# Item_Weight
train['Item_Weight'].describe()

# Outlet Size
train['Outlet_Size'].value_counts()

# Item_Fat_Content
prev = train['Item_Fat_Content'].value_counts()
train['Item_Fat_Content'].replace({'reg':'Regular',
                                   'LF':'Low Fat',
                                   'low fat':'Low Fat'},
                                  inplace=True)
later = train['Item_Fat_Content'].value_counts()

# Item_Visibility
train['Item_Visibility'].describe()

# Item_Type
train['Item_Type'].value_counts()

# Imputing Item Weights
items_trn = train[['Item_Identifier', 
               'Item_Weight']].sort_values(by='Item_Identifier')
items_trn = items_trn[items_trn['Item_Weight'].notna()]

items_tst = test[['Item_Identifier', 
               'Item_Weight']].sort_values(by='Item_Identifier')
items_tst = items_tst[items_tst['Item_Weight'].notna()]

items = pd.concat([items_trn, items_tst])
items = items.drop_duplicates()
train.drop('Item_Weight', axis=1, inplace=True)

train_wt = train.merge(items, how='inner', 
                       on='Item_Identifier')

# Imputing Outlet Size
outlets_trn = train[['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']]
outlets_tst = test[['Outlet_Identifier','Outlet_Establishment_Year',
                'Outlet_Size', 'Outlet_Location_Type','Outlet_Type']]

outlets = pd.concat([outlets_trn, outlets_tst])
outlets.drop_duplicates(inplace=True)

train_wt['Outlet_Size'].fillna(value="Small", inplace=True)
train_wt.isnull().sum()

X_train = train_wt.drop(['Item_Identifier','Item_Outlet_Sales',
                         'Outlet_Identifier'],axis=1)
y_train = train_wt['Item_Outlet_Sales']


ohe = OneHotEncoder(sparse_output=False, 
                    handle_unknown='ignore').set_output(transform='pandas')

col_trn = make_column_transformer(
    (ohe, make_column_selector(dtype_include=object)),
    ('passthrough',make_column_selector(dtype_exclude=object)),
           verbose_feature_names_out=False)
col_trn = col_trn.set_output(transform='pandas')
gbm = xgb.XGBRegressor(random_state=24)
pipe = Pipeline([('TRNSF',col_trn),('ML',gbm)])

params = {'ML__learning_rate':np.linspace(0.001, 1, 5),
          'ML__max_depth': [2,3,4,5,6],
          'ML__n_estimators':[50,100,150]}

kfold = KFold(n_splits=5, shuffle=True, random_state=24)

gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='r2', verbose=3)

gcv.fit(X_train, y_train)

best_model = gcv.best_estimator_
########## Test set Operations
test = pd.read_csv("test_AbJTz2l.csv")
test.drop('Item_Weight', axis=1, inplace=True)

test_wt = test.merge(items, how='inner', 
                       on='Item_Identifier')
test_wt['Outlet_Size'].fillna(value="Small", inplace=True)
test_wt.isnull().sum()

X_test = test_wt.drop(['Item_Identifier',
                         'Outlet_Identifier'],axis=1)
y_pred = best_model.predict(X_test)

y_pred[y_pred<0] = 33.29

test_wt['Sales'] = y_pred
submit = pd.read_csv("sample_submission_8RXa3c6.csv")

test_submit = test_wt[['Item_Identifier','Outlet_Identifier',
                           'Sales']]

submission = submit.merge(test_submit, on=['Item_Identifier',
                                           'Outlet_Identifier'])
submission['Item_Outlet_Sales'] = submission['Sales']
submission.drop('Sales', axis=1, inplace=True)

submission.to_csv("XGB_reg.csv", index=False)


