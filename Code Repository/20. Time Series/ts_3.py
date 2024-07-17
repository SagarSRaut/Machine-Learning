from statsmodels.tsa.api import SimpleExpSmoothing
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from numpy import sqrt 
from sklearn.metrics import mean_squared_error as mse

######### Shampoo
shamp = pd.read_csv("sales-of-shampoo-over-a-three-ye.csv")

#vals = shamp.values[:,0]
shamp.plot()
plt.title("Monthly Shampoo Sales")
#plt.text(1,vals[0],str(vals[0]) )
plt.show()

y = shamp['Sales of shampoo over a three year period']
y_train = y[:-6]
y_test = y[-6:]
alpha = 0.4
# Simple Exponential Smoothing

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
alpha = 0.25
beta = 0.25
gamma = 0.01
hw_add = ExponentialSmoothing(y_train, seasonal_periods=12, 
                 trend='add', seasonal='add',use_boxcox=0.5)
fit1 = hw_add.fit(smoothing_level=alpha, 
       smoothing_trend=beta,smoothing_seasonal=gamma)
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
alpha = 0.45
beta = 0.4
gamma = 0.3
hw_mul = ExponentialSmoothing(y_train, seasonal_periods=12,
         trend='add', seasonal='mul',use_boxcox=0.4)
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


################# Change the time series to Quarterly

shamp = pd.read_csv("sales-of-shampoo-over-a-three-ye.csv",
                    index_col=0)
shamp.index = pd.to_datetime( shamp.index ).to_period("M")

shamp.plot()
plt.title("Monthly Shampoo Sales")
plt.show()

shamp_qtr  = shamp.resample('Q').sum() 
shamp_qtr.index.rename('Quarter', inplace=True)
shamp_qtr.plot()
plt.title("Quarterly Shampoo Sales")
plt.xlabel("Quarters")
plt.show()

y = shamp_qtr['Sales of shampoo over a three year period']
y_train = y[:-3]
y_test = y[-3:]
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
