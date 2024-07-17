import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\Glass.csv")
le = LabelEncoder()
y = le.fit_transform(glass['Type'])
X = glass.drop('Type', axis=1)
print(le.classes_)

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

#### unlabelled data
tst_glass = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Glass Identification\tst_Glass.csv")
best_model = gcv.best_estimator_
predictions = best_model.predict(tst_glass)
print(predictions)
pred_type = le.inverse_transform(predictions)
print(pred_type)
