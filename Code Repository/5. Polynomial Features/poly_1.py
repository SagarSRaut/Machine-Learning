import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

pizza = pd.read_csv("pizza.csv")

X = pizza[['Promote']]
y = pizza['Sales']

poly = PolynomialFeatures(degree=2).set_output(transform="pandas")
X_poly = poly.fit_transform(X)

lr = LinearRegression()
lr.fit(X_poly, y)

print(lr.coef_, lr.intercept_)

#########
insure = pd.read_csv("Insure_Auto.csv", index_col=0)
X = insure.drop('Operating_Cost', axis=1)
y = insure['Operating_Cost']

poly = PolynomialFeatures(degree=2).set_output(transform="pandas")
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, 
                                   test_size = 0.3, 
                                   random_state=24)

lr = LinearRegression()
lr.fit(X_train, y_train)


##### Boston
boston = pd.read_csv("Boston.csv")
y = boston['medv']
X = boston.drop('medv', axis=1)

poly = PolynomialFeatures(degree=2).set_output(transform="pandas")
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, 
                                   test_size = 0.3, 
                                   random_state=24)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))