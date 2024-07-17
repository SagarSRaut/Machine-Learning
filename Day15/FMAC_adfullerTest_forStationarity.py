import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Day15")

df = pd.read_csv('FMAC-HPI_24420.csv')
df.head()

#########stationarity test

result = adfuller(df['SA Value'], maxlag=10)
print('P-vaue: ', result[1])


def stationarity(n):
    if n < 0.05:
        return 'Time series is stationary'
    else:
        return 'Time series is not stationary'
print(stationarity(result[1]))



#### applying differencing 1st order, 2nd order.... to achieve stationarity

diff_1 = df['SA Value'].diff(1)
diff_1.dropna(inplace=True)
result_diff = adfuller(diff_1, maxlag=10)
print('P-vaue: ', result_diff[1])
      
print(stationarity(result_diff[1]))


################################################################



result_nsa = adfuller(df['NSA Value'], maxlag=10)
print('P-vaue: ', result_nsa[1])


def stationarity(n):
    if n < 0.05:
        return 'Time series is stationary'
    else:
        return 'Time series is not stationary'
print(stationarity(result_nsa[1]))

#### applying differencing 1st order, 2nd order.... to achieve stationarity

diff_1 = df['NSA Value'].diff(1)
diff_1.dropna(inplace=True)
result_diff = adfuller(diff_1, maxlag=10)
print('P-vaue: ', result_diff[1])
      
print(stationarity(result_diff[1]))

#### because the first order differencing did not work, we are applying 2nd order

diff_2 = diff_1.diff(1)
diff_2.dropna(inplace=True)
result_diff2 = adfuller(diff_2, maxlag=10)
print('P-vaue: ', result_diff2[1])
      
print(stationarity(result_diff2[1]))







