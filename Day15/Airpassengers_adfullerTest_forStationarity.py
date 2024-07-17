import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Day15")

df = pd.read_csv('AirPassengers.csv')
df.head()

#########stationarity test

result = adfuller(df['Passengers'], maxlag=10)
print('P-vaue: ', result[1])


def stationarity(n):
    if n < 0.05:
        return 'Time series is stationary'
    else:
        return 'Time series is not stationary'

print(stationarity(result[1]))


###################### finding the autocorrelation

plot_acf(df['Passengers'], lags=30, alpha=None)
plt.show()


plot_acf(df['Passengers'], lags=30)
plt.show()

ac_vals = sm.tsa.acf(df['Passengers'])
print(ac_vals)