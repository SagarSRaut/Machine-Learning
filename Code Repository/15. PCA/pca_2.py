import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
import numpy as np 

brupt = pd.read_csv("C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy\Bankruptcy.csv")
X = brupt.drop(['NO','D'], axis=1)
y = brupt['D']
lr = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                   test_size = 0.3, 
                                   random_state=24,
                                   stratify=y)

scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA(n_components=11).set_output(transform='pandas')

pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('LR',lr)])
pipe.fit(X_train, y_train)
print(np.cumsum(prcomp.explained_variance_ratio_ * 100)) 

y_pred = pipe.predict(X_test)
y_pred_prob = pipe.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(log_loss(y_test, y_pred_prob))

############### Grid Search CV ###################
print(pipe.get_params())
params = {'PCA__n_components': np.arange(6,12),
          'LR__C': np.linspace(0.001, 3, 5)}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold ,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############ Glass Identification ####################
glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\Glass.csv")
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis=1)

scaler = StandardScaler().set_output(transform='pandas')
prcomp = PCA().set_output(transform='pandas')
pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('LR',lr)])
params = {'PCA__n_components': [5,6,7,8,9],
          'LR__C': np.linspace(0.001, 3, 5),
          'LR__multi_class':['ovr','multinomial']}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold ,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

################# NB #######################
nb = GaussianNB()
pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('NB',nb)])
params = {'PCA__n_components': [5,6,7,8,9]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold ,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

##################### RF ###########################
rf = RandomForestClassifier(random_state=24)
pipe = Pipeline([('SCL',scaler),('PCA',prcomp),('RF',rf)])
params = {'PCA__n_components': [5,6,7,8,9]}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
gcv = GridSearchCV(pipe, param_grid=params, cv=kfold ,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)

############## t-SNE #########################
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import seaborn as sns
tsne = TSNE(n_components=2, random_state=24, 
            perplexity=5).set_output(transform='pandas')
embed_tsne = tsne.fit_transform(X)

embed_tsne['Type'] = le.fit_transform(glass['Type'])
embed_tsne['Type'] = embed_tsne['Type'].astype(str)
sns.scatterplot(data=embed_tsne, x='tsne0', y='tsne1',
                hue='Type')
plt.show()


params = {}
kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
gcv = GridSearchCV(lr, param_grid=params, cv=kfold ,
                   scoring='neg_log_loss')
embed_tsne = tsne.fit_transform(X)
gcv.fit(embed_tsne, y)
print(gcv.best_params_)
print(gcv.best_score_)
