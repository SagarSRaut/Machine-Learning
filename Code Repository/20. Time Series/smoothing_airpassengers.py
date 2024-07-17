from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from numpy import sqrt 
from sklearn.metrics import mean_squared_error as mse

df = pd.read_csv("AirPassengers.csv")
df.head()

y_train = df['Passengers'][:-12]
y_test = df['Passengers'][-12:]

alpha = 0.4
# Simple Exponential Smoothing
from statsmodels.tsa.api import SimpleExpSmoothing
ses = SimpleExpSmoothing(y_train)
fit1 = ses.fit(smoothing_level=alpha)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.legend(loc='best')
plt.show()

#print(fit1.params)
print("RMSE =",sqrt(mse(y_test, fcast1)))

# Holt's Linear Method
alpha = 0.4
beta = 0.8
### Linear Trend
from statsmodels.tsa.api import Holt
holt = Holt(y_train)
fit1 = holt.fit(smoothing_level=alpha,
            smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')

plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

error = round(sqrt(mse(y_test, fcast1)),2)
print("MSE =", error)


# Holt's Exponential Method
alpha = 0.4
beta = 0.8
from statsmodels.tsa.api import Holt
holt = Holt(y_train, exponential=True)
fit1 = holt.fit(smoothing_level=alpha,
            smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')

plt.title("Holt's Exponential Trend")
plt.legend(loc='best')
plt.show()

error = round(sqrt(mse(y_test, fcast1)),2)
print("MSE =", error)


# Holt-Winters' Method

########### Additive #####################
from statsmodels.tsa.api import ExponentialSmoothing
alpha = 0.15
beta = 0.55
gamma = 0.55
hw_add = ExponentialSmoothing(y_train, seasonal_periods=12, 
                 trend='add', seasonal='add')
fit1 = hw_add.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(10,400, "RMSE="+str(error))
plt.title("HW Additive Trend and Seasonal Method")
plt.legend(loc='best')
plt.show()


########### Multiplicative #####################
alpha = 0.1
beta = 0.95
gamma = 0.3
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12,
         trend='add', seasonal='mul')
fit1 = hw_mul.fit(smoothing_level=alpha, 
                    smoothing_trend=beta,
                    smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(10,400, "RMSE="+str(error))
plt.title("HW Additive Trend and Multiplicative Seasonal Method")
plt.legend(loc='best')
plt.show()
