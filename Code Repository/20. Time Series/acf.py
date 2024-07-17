import os
os.chdir("C:/Training/Academy/Statistics (Python)/Datasets")
import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("AusGas.csv")
df.head()

plot_acf(df['GasProd'], lags=30,alpha=None)
plt.show()

plot_acf(df['GasProd'], lags=30)
plt.show()

acf_vals = sm.tsa.acf(df['GasProd'])
print(acf_vals)

###### Stationarity Test
result = adfuller(df['GasProd'], maxlag=10)
print("P-Value =", result[1])
if result[1] < 0.05:
    print("Time Series is Stationary")
else:
    print("Time Series is not Stationary")