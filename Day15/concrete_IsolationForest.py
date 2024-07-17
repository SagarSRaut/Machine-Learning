import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import numpy as np

concrete = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Concrete Strength\Concrete_Data.csv')
concrete.head()

X = concrete.drop('Strength', axis=1)  #take all the columns upto the salary column
y = concrete['Strength'] #take the first column
print(y)


lr=LinearRegression()



kfold=KFold(n_splits=5,shuffle=True,random_state=24)

results= cross_val_score(lr,X,y,cv=kfold,scoring='r2')
print(results.mean())


############ with anomaly detection

clf = IsolationForest(contamination=0.05, random_state=24)
clf.fit(concrete)
predictions = clf.predict(concrete)

print("%age of outliers="+ str((predictions<0).mean()*100)+ "%")
abn_ind = np.where(predictions > 0)
print("Inliers:")
print(concrete.index[abn_ind])
inliers = concrete.index[abn_ind]



# df_cleaned = concrete.drop(abn_ind, axis=0)
con_cleaned = concrete.iloc[np.array(abn_ind[0]),:]

X = con_cleaned.drop('Strength', axis=1)  #take all the columns upto the salary column
y = con_cleaned['Strength']


lr_o=LinearRegression()



kfold=KFold(n_splits=5,shuffle=True,random_state=24)

results= cross_val_score(lr_o,X,y,cv=kfold,scoring='r2')
print(results.mean())

# if concrete.index in abn_index:
#     print(pd.Series(concrete))
