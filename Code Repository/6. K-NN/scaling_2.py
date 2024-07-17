import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

kyp = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Kyphosis\Kyphosis.csv")
le = LabelEncoder()
y = le.fit_transform(kyp['Kyphosis'])
X = kyp.drop('Kyphosis', axis=1)

knn = KNeighborsClassifier(n_neighbors=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.3, 
                                                    random_state=24,
                                                    stratify=y)
## w/o scaling
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

y_pred_prob = knn.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]))
print(log_loss(y_test, y_pred_prob))

### Standard
std_scaler = StandardScaler()

pipe_std = Pipeline([('SCL', std_scaler),('KNN',knn)])
pipe_std.fit(X_train, y_train)
y_pred = pipe_std.predict(X_test)

y_pred_prob = pipe_std.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]))
print(log_loss(y_test, y_pred_prob))

### Min Max Scaler
mm_scaler = MinMaxScaler()

pipe_mm = Pipeline([('SCL', mm_scaler),('KNN',knn)])
pipe_mm.fit(X_train, y_train)
y_pred = pipe_mm.predict(X_test)

y_pred_prob = pipe_mm.predict_proba(X_test)
print(accuracy_score(y_test, y_pred))
print(roc_auc_score(y_test, y_pred_prob[:,1]))
print(log_loss(y_test, y_pred_prob))

############# Grid Search ##########################
kfold = StratifiedKFold(n_splits=5, shuffle=True,
                        random_state=24)
std_scaler = StandardScaler()
mm_scaler = MinMaxScaler()
knn = KNeighborsClassifier()
pipe = Pipeline([('SCL', None),('KNN',knn)])
params = {'KNN__n_neighbors':[1,2,3,4,5,6,7,8,9,10],
          'SCL':[std_scaler, mm_scaler, None]}

gcv = GridSearchCV(pipe, param_grid=params, cv=kfold,
                   scoring='neg_log_loss')
gcv.fit(X, y)
print(gcv.best_params_)
print(gcv.best_score_)




