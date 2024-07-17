from sklearn.metrics import mean_squared_error as mse 
import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from numpy import sqrt 

gdp = pd.read_csv("gdp-per-capita-ppp-constant-2011.csv")
y_gdp = gdp['GDP_per_capita']
y_trn_gdp = y_gdp.iloc[:-4]
y_tst_gdp = y_gdp.iloc[-4:]

# Holt's Linear Method
alpha = 0.15
beta = 1
from statsmodels.tsa.api import Holt
holt = Holt(y_trn_gdp)
fit1 = holt.fit(smoothing_level=alpha,
            smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_tst_gdp))
# plot
y_trn_gdp.plot(color="blue", label='Train')
y_tst_gdp.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()

print(sqrt(mse(y_tst_gdp, fcast1)))

# Holt's Exponential Trend Method
alpha = 0.1
beta = 1
from statsmodels.tsa.api import Holt
holt = Holt(y_trn_gdp, exponential=True)
fit1 = holt.fit(smoothing_level=alpha,
            smoothing_trend=beta)
fcast1 = fit1.forecast(len(y_tst_gdp))
# plot
y_trn_gdp.plot(color="blue", label='Train')
y_tst_gdp.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.title("Holt's Exponential Trend")
plt.legend(loc='best')
plt.show()

print(sqrt(mse(y_tst_gdp, fcast1)))

# auto - tune
holt = Holt(y_trn_gdp, exponential=True)
fit1 = holt.fit()
fcast1 = fit1.forecast(len(y_tst_gdp))
# plot
y_trn_gdp.plot(color="blue", label='Train')
y_tst_gdp.plot(color="pink", label='Test')
fcast1.plot(color="purple", label='Forecast')
plt.title("Holt's Linear Trend")
plt.legend(loc='best')
plt.show()
print(sqrt(mse(y_tst_gdp, fcast1)))
print(fit1.params)

###  Damped Trend
alpha = 0.8
beta = 0.02
phi = 0.1
add_damp = Holt(y_trn_gdp, damped_trend=True,
                exponential=True)
fit3 = add_damp.fit(smoothing_level=alpha, 
                    smoothing_trend=beta, 
                    damping_trend=phi)
fcast3 = fit3.forecast(len(y_tst_gdp))
# plot
y_trn_gdp.plot(color="blue", label='Train')
y_tst_gdp.plot(color="pink", label='Test')
fcast3.plot(color="purple", label='Forecast')
plt.title("Additive Damped Trend")
plt.legend(loc='best')
plt.show()
print(sqrt(mse(y_tst_gdp, fcast1)))








