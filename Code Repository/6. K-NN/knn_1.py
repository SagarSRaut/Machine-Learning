import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

knn = KNeighborsClassifier(n_neighbors=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

y_pred_prob = knn.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]))
print(log_loss(y_test, y_pred_prob))

################## Grid Search #################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
params = {'n_neighbors':[1,2,3,4,5]}
knn = KNeighborsClassifier()
gcv = GridSearchCV(knn, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)


