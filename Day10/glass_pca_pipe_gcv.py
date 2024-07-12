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
from sklearn.tree import DecisionTreeClassifier, plot_tree, DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier, BaggingClassifier,StackingClassifier,GradientBoostingClassifier,RandomForestClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

glass = pd.read_csv(r'C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Glass Identification\Glass.csv')
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop(['Type'], axis=1)
print(le.classes_)

lr = LogisticRegression()
nb = GaussianNB()
rfc=RandomForestClassifier(random_state=24)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)

scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components=7).set_output(transform='pandas')

pipe = Pipeline([('SCL', scaler),('PCA', prcomp), ('RFC', rfc)])
pipe.fit(X_train, y_train)
print(np.cumsum(prcomp.explained_variance_ratio_*100))

y_pred = pipe.predict(X_test)
y_pred_prob = pipe.predict_proba(X_test)

print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))




###########################grid search######################
print(pipe.get_params())
params= {'PCA__n_components': [5,6,7,8,9]}
kfold= StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
gcv=GridSearchCV(pipe, param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(X,y)
print(gcv.best_params_)
print(gcv.best_score_)


############# t-SNE #########################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
tsne = TSNE(n_components=2, random_state=24,
            perplexity=20).set_output(transform='pandas')

embed_tsne = tsne.fit_transform(X)

embed_tsne['Type'] = le.fit_transform(glass['Type'])
embed_tsne['Type'] = embed_tsne['Type'].astype(str)


sns.scatterplot(data=embed_tsne, x='tsne0',y='tsne1', hue ='Type')
plt.show()

 
params={}
kfold= StratifiedKFold(n_splits=5,shuffle=True,random_state=24)
gcv=GridSearchCV(lr, param_grid=params,cv=kfold,scoring='neg_log_loss')
gcv.fit(embed_tsne,y)
print(gcv.best_params_)
print(gcv.best_score_)

