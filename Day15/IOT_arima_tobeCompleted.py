from pmdarima.arima import auto_arima
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import os

os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Day15\archive (1)")


iot = pd.read_csv("train_ML_IOT.csv", index_col=0)
iot['DateTime'] = pd.to_datetime(iot['DateTime'])
iot.set_index('Datetime', inplace=True)

iot_m = iot[iot['Junction'] == 1] 
iot_v = iot_m['Vehicles']

iot_month = iot_v.resample('M').sum()

iot_month = iot.resample('M').sum()
iot_month.index.rename('Month', inplace= True)
iot_month.plot()
plt.title('Monthly shampoo sales')
plt.xlabel('Months')
plt.show()



y = shamp_qtr['Junction']
y_train = y[:-3]
y_test = y[-3:]
