import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder , StandardScaler,MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score,KFold
from sklearn.linear_model import LinearRegression, Ridge,ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier


kyp = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Kyphosis\Kyphosis.csv')
kyp.head()


#Performs one-hot encoding using pandas
le = LabelEncoder()
y=le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis',axis=1)


# min_samples_split is the minimum number of samples on which the tree will split
svm_l = SVC(kernel='linear', probability=True, random_state=24)
std_scaler = StandardScaler()

pipe_l = Pipeline([('SCL', std_scaler), ('SVM', svm_l)])


svm_r = SVC(kernel='rbf', probability=True, random_state=24)

pipe_r = Pipeline([('SCL', std_scaler), ('SVM', svm_r)])

lr = LogisticRegression()
lda = LinearDiscriminantAnalysis()
dtc = DecisionTreeClassifier(random_state=24)

voting = VotingClassifier([('LR',lr),('SVML',pipe_l),('SVMR',pipe_r), 
                           ('LDA', lda), ('TREE',dtc)], voting='soft')



X_train, X_test, y_train, y_test = train_test_split(X, y,
                                   stratify=y,
                                   random_state=24, test_size=0.3)


voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)
print(accuracy_score(y_test, y_pred))

y_pred_prob = voting.predict_proba(X_test)[:,1]
print(log_loss(y_test, y_pred_prob))


#############################################################################

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
print(voting.get_params())

params = {'SVML__SVM__C': np.linspace(0.001,3,5),
          'TREE__max_depth':[None,2,3],
          'SVMR__SVM__gamma':np.linspace(0.001,3,5),
          'SVMR__SVM__C': np.linspace(0.001,3,5),
          'LR__C': np.linspace(0.001,3,5)}

gcv = GridSearchCV(voting,param_grid=params,cv=kfold, 
                   scoring='neg_log_loss', n_jobs=-1)

gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)




