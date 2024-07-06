
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge,ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.compose import make_column_selector, make_column_transformer


#Performing encoding on the label
housing = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\Housing.csv")

ohc = OneHotEncoder() 
scaler=StandardScaler()

ct = make_column_transformer((ohc, make_column_selector(dtype_include=object)), ('passthrough', 
                             make_column_selector(dtype_include=['int64','float64'])))
dum_pd = ct.fit_transform(housing)
print(ct.get_feature_names_out())


#################################

ohc = OneHotEncoder(sparse_output=False,drop='first')
ct = make_column_transformer((ohc, make_column_selector(dtype_include=object)), ('passthrough', 
                             make_column_selector(dtype_include=['int64','float64'])),verbose_feature_names_out = False).set_output(transform='pandas')
dum_pd = ct.fit_transform(housing)
print(ct.get_feature_names_out())