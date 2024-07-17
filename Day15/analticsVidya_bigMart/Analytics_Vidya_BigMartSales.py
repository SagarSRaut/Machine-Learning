import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import seaborn as sns 
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.impute import SimpleImputer 


os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Day15\analticsVidya_bigMart")

train = pd.read_csv("train_v9rqX0R.csv")
X_train = train.drop(['Outlet_Identifier', 'Item_Identifier','Item_Outlet_Sales'], axis=1)
X_train.info()

############### Cleaning 

X_train.isna().sum()

X_train.value_counts()












y_train = train['Item_Outlet_Sales']


test = pd.read_csv("test_AbJTz2l.csv")
test.drop(['Outlet_Identifier', 'Item_Identifier'], axis=1, inplace=True)
test.info()


simple_obj = SimpleImputer(strategy='most_frequent')

ct_imp_1 = make_column_transformer((simple_obj,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_exclude=object)), 
       verbose_feature_names_out=False ).set_output(transform='pandas')


simple_num = SimpleImputer(strategy='median')

ct_imp_2 = make_column_transformer((simple_num,
       make_column_selector(dtype_include=['int','float'])), 
       ("passthrough",
        make_column_selector(dtype_include=object)), 
       verbose_feature_names_out=False ).set_output(transform='pandas')

##############################################################################

ct_imp_1.fit_transform(X_train)
ct_imp_2.fit_transform(X_train)

##############################################################################

ct_imp_1.transform(test)
ct_imp_2.transform(test)

############### Transforming train set


ohc = OneHotEncoder(sparse_output=False, drop='first')
ct_train = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_exclude=object)), 
       verbose_feature_names_out=False ).set_output(transform='pandas')
train_t = ct_train.fit_transform(train)
train_t.info()

################Transforming the test set



ct_test = make_column_transformer((ohc,
       make_column_selector(dtype_include=object)),
       ("passthrough",
        make_column_selector(dtype_exclude=object)), 
       verbose_feature_names_out=False ).set_output(transform='pandas')
test_t = ct_test.fit_transform(train)
test_t.info()














