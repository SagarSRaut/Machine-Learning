import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import plot_tree 

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)

dtc = DecisionTreeClassifier(random_state=24,
                             min_samples_leaf=14)
dtc.fit(X_train, y_train)

plt.figure(figsize=(25,20))
plot_tree(dtc,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=25)
plt.show() 

y_pred = dtc.predict(X_test)
print(accuracy_score(y_test, y_pred))

#############################################
params = {'min_samples_split':[2,4,6,10,20],
          'min_samples_leaf':[1,5,10,15],
          'max_depth':[None, 4, 3, 2]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(dtc, param_grid=params, 
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_tree = gcv.best_estimator_
plt.figure(figsize=(25,20))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['0','1'],
               filled=True,fontsize=30)
plt.title("Best Tree")
plt.show() 

################ Glass ######################
glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\Glass.csv")
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis=1)
print(le.classes_)

params = {'min_samples_split':np.arange(2,35,5),
          'min_samples_leaf':np.arange(1, 35, 5),
          'max_depth':[None, 4, 3, 2]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(dtc, param_grid=params, 
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_tree = gcv.best_estimator_
plt.figure(figsize=(30,20))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=list(le.classes_),
               filled=True,fontsize=25)
plt.title("Best Tree")
plt.show() 

print(best_tree.feature_importances_)

df_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':best_tree.feature_importances_})

plt.bar(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importances")
plt.show()


##################################################
m_left, m_right = 183, 31
g_left, g_right = 0.679, 0.287
m = 214

ba_split = (m_left/m)*g_left + (m_right/m)*g_right
ba = 0.737 - ba_split

m_left, m_right = 183, 31
g_left, g_right = 0.679, 0.287
m = 214

ba_split = (m_left/m)*g_left + (m_right/m)*g_right
ba_reduction = 0.737 - ba_split


m_left, m_right = 113, 70
g_left, g_right = 0.6, 0.584
m = 183

al_split = (m_left/m)*g_left + (m_right/m)*g_right
al_reduction = 0.679 - al_split

################### HR Analytics #############
hr = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\human-resources-analytics\HR_comma_sep.csv")
dum_hr = pd.get_dummies(hr, drop_first=True)
X = dum_hr.drop('left', axis=1)
y = dum_hr['left']

params = {'min_samples_split':np.arange(2,35,5),
          'min_samples_leaf':np.arange(1, 35, 5),
          'max_depth':[None, 4, 3, 2]}
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
dtc = DecisionTreeClassifier(random_state=24)
gcv = GridSearchCV(dtc, param_grid=params, verbose=3,
                   cv=kfold, scoring="neg_log_loss")
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

best_tree = gcv.best_estimator_
plt.figure(figsize=(50,20))
plot_tree(best_tree,feature_names=list(X.columns),
               class_names=['Not Left','Left'],
               filled=True,fontsize=25)
plt.title("Best Tree")
plt.show() 

print(best_tree.feature_importances_)

df_imp = pd.DataFrame({'Feature':list(X.columns),
                       'Importance':best_tree.feature_importances_})

plt.barh(df_imp['Feature'],
        df_imp['Importance'])
plt.title("Feature Importances")
plt.show()

