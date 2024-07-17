import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression 

pizza = pd.read_csv("pizza.csv")

xi = pizza['Promote']
yi = pizza['Sales']
n = pizza.shape[0]

xbar = np.mean(xi)
ybar = np.mean(yi)

m_xi_yi = np.sum(xi*yi)/n
m_xi_2 = np.sum(xi**2)/n

b1 = (m_xi_yi - (xbar*ybar))/(m_xi_2 - (xbar**2))
b0 = ybar - (b1*xbar)

##########################################
X = pizza[['Promote']]
y = pizza['Sales']

lr = LinearRegression()
lr.fit(X, y)

print(lr.intercept_, lr.coef_)

#####################################
A = np.array([[2,1],[3,2]])
b = np.array([4,7])

print(np.linalg.solve(A,b))

##################################
insure = pd.read_csv("Insure_Auto.csv",
                     index_col=0)

x1 = insure['Home']
x2 = insure['Automobile']
y = insure['Operating_Cost']

x1_sum = np.sum(x1)
x1_2_sum = np.sum(x1*x1)
x1_x2_sum = np.sum(x1*x2)
x1_y = np.sum(x1*y)

x2_sum = np.sum(x2)
x2_2_sum = np.sum(x2*x2)
x2_y = np.sum(x2*y)
x1_bar = np.mean(x1)
x2_bar = np.mean(x2)
y_bar = np.mean(y)

p = np.array([[x1_sum, x1_2_sum, x1_x2_sum], 
              [x2_sum, x1_x2_sum, x2_2_sum],
              [1, x1_bar, x2_bar]])
q = np.array([x1_y, x2_y, y_bar])

x = np.linalg.solve(p,q)
print(x)

#### Verifying with sklearn
X = insure[['Home','Automobile']]
y = insure['Operating_Cost']

lr = LinearRegression()
lr.fit(X, y)

print(lr.intercept_, lr.coef_)
