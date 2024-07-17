import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import log_loss

glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\Glass.csv")
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis=1)
print(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
lda = LinearDiscriminantAnalysis()


lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#################################################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
results = cross_val_score(lda, X, y, scoring='neg_log_loss')
print(results.mean())

##########################################################
brupt = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Bankruptcy\Bankruptcy.csv",
                    index_col=0)
X = brupt.drop('D', axis=1)
y = brupt['D']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
y_pred = lda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = lda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred = qda.predict(X_test)
print(accuracy_score(y_test, y_pred))
y_pred_prob = qda.predict_proba(X_test)
print(log_loss(y_test, y_pred_prob))

#################################################
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
results = cross_val_score(lda, X, y, scoring='neg_log_loss',
                          cv=kfold)
print(results.mean())

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
results = cross_val_score(qda, X, y, scoring='neg_log_loss',
                         cv=kfold)
print(results.mean())

#########################################################
satellite = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Satellite Imaging\Satellite.csv",
                        sep=";")
le = LabelEncoder()
y = le.fit_transform(satellite['classes'])
X = satellite.drop('classes', axis=1)
print(le.classes_)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
results = cross_val_score(lda, X, y, scoring='neg_log_loss',
                          cv=kfold)
print(results.mean())

kfold = StratifiedKFold(n_splits=5, shuffle=True, 
                        random_state=24)
results = cross_val_score(qda, X, y, scoring='neg_log_loss',
                         cv=kfold)
print(results.mean())

