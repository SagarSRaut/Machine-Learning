from sklearn.metrics import mean_squared_error as mse 
import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from numpy import sqrt 

df = pd.read_csv("monthly-milk-production-pounds-p.csv")
df.head()

df.plot()
plt.show()

y = df['Milk']


#### Centered MA
fcast = y.rolling(3,center=True).mean()
plt.plot(y, label='Original Data')
plt.plot(fcast, label='Centered Moving Average')
plt.legend(loc='best')
plt.show()



y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
plt.plot(y_train, label='Train',color='blue')
plt.plot(y_test, label='Test',color='orange')
plt.legend(loc='best')
plt.show()

#### Trailing Rolling Mean
fcast = y_train.rolling(4,center=False).mean()
lastMA = fcast.iloc[-1]
fSeries = pd.Series(lastMA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast, fSeries],
                     ignore_index=True)
plt.plot(y_train, label='Train',color='blue')
plt.plot(MA_fcast, label='Moving Average Forecast',
         color='red')
plt.plot(y_test, label="Test")
plt.legend(loc='best')
plt.show()

print("RMSE =",sqrt(mse(y_test, fSeries)))

y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
alpha = 0.1
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

print(fit1.params)
print("RMSE =",sqrt(mse(y_test, fcast1)))


# Holt's Linear Method
alpha = 0.5
beta = 0.02
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
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()


### auto-tune
holt = Holt(y_train)
fit1 = holt.fit()
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()
print(fit1.params)

#print("MSE =",mse(y_test, fcast1))


# Holt's Exponential Method
alpha = 0.8
beta = 0.02

holt = Holt(y_train, exponential=True)
fit1 = holt.fit(smoothing_level=alpha, 
                smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_test))

# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("Holt's Exponential Trend")
plt.legend(loc='best')
plt.show()

#print("MSE =",mse(y_test, fcast1))


### Additive Damped Trend
alpha = 0.8
beta = 0.02
phi = 0.1
add_damp = Holt(y_train, damped_trend=True)
fit3 = add_damp.fit(smoothing_level=alpha, 
                    smoothing_trend=beta, 
                    damping_trend=phi)
fcast3 = fit3.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast3)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("Additive Damped Trend")
plt.legend(loc='best')
plt.show()

### Multiplicative Damped Trend
alpha = 0.8
beta = 0.02
phi = 0.1
mult_damp = Holt(y_train, damped_trend=True, 
                exponential=True)
fit3 = mult_damp.fit(smoothing_level=alpha, 
                    smoothing_trend=beta, 
                    damping_trend=phi)
fcast3 = fit3.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast3)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("Multiplicative Damped Trend")
plt.legend(loc='best')
plt.show()

# Holt-Winters' Method

######################## Additive ########################
from statsmodels.tsa.api import ExponentialSmoothing
alpha, beta, gamma = 0.8, 0.02, 0.1
hw_add = ExponentialSmoothing(y_train, seasonal_periods=12, 
         trend='add', seasonal='add')
fit1 = hw_add.fit(smoothing_level=alpha, 
       smoothing_trend=beta,smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("HW Additive Trend and Seasonal Method")
plt.legend(loc='best')
plt.show()



########### Multiplicative #####################
alpha, beta, gamma = 0.8, 0.02, 0.1
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, 
                            trend='add', seasonal='mul')
fit1 = hw_mul.fit(smoothing_level=alpha, 
        smoothing_trend=beta, smoothing_seasonal=gamma)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("HW Additive Trend and Multiplicative Seasonal Method")
plt.legend(loc='best')
plt.show()


########### Seasonal Additive & Damped #####################
alpha, beta, gamma, phi = 0.8, 0.02, 0.1, 0.1

hw_add = ExponentialSmoothing(y_train, seasonal_periods=12, 
         trend='add', seasonal='add',damped_trend=True)
fit1 = hw_add.fit(smoothing_level=alpha, 
       smoothing_trend=beta,smoothing_seasonal=gamma,
       damping_trend=phi)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("Damped HW Additive Trend and Seasonal Method")
plt.legend(loc='best')
plt.show()


########### Seasonal Multiplicative & Damped #####################
alpha, beta, gamma, phi = 0.8, 0.02, 0.1, 0.1
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12, 
         trend='add', seasonal='mul',damped_trend=True)
fit1 = hw_mul.fit(smoothing_level=alpha, 
       smoothing_trend=beta,smoothing_seasonal=gamma,
       damping_trend=phi)
fcast1 = fit1.forecast(len(y_test))
# plot
y_train.plot(color="blue", label='Train')
y_test.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
error = round(sqrt(mse(y_test, fcast1)),2)
plt.text(100,600, "RMSE="+str(error))
plt.title("Damped HW Additive Trend and Multiplicative Seasonal Method")
plt.legend(loc='best')
plt.show()

# ################## sktime #################################

# from sktime.forecasting.ets import AutoETS
# forecaster = AutoETS(auto=True, n_jobs=-1, sp=12)
# forecaster.fit(y_train)
# print(forecaster.summary())
# y_pred = forecaster.predict(fh=[1,2,3])
