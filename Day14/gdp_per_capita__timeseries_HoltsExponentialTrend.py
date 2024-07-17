import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

gdp= pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\gdp-per-capita-ppp-constant-2011.csv")
gdp.head()
y_gdp=gdp['GDP_Per_capita']
y_trn_gdp = y_gdp[:-4]
y_tst_gdp = y_gdp[-4:]

################## holt 
alpha=1
beta=0.1
        

from statsmodels.tsa.api import SimpleExpSmoothing,Holt
holt=Holt(y_trn_gdp, exponential=True)

fit1=holt.fit(smoothing_level=alpha,smoothing_trend=beta)
fcast1 =fit1.forecast(len(y_tst_gdp))
 
### plotting
y_trn_gdp.plot(color="blue",label='Train')
y_tst_gdp.plot(color="pink",label='Test')
fcast1.plot(color="purple",label='Forecast')
plt.title("Holt's exponential trend")
plt.legend(loc='best')
plt.show()

print(np.sqrt(mean_squared_error(y_tst_gdp, fcast1)))

 
###### Autotuning - not mentioning the values of alpha and beta explicitly ###############
######## Exponential = True
holt=Holt(y_trn_gdp, exponential=True)

fit1=holt.fit()
fcast1 =fit1.forecast(len(y_tst_gdp))
  
### plotting
y_trn_gdp.plot(color="blue",label='Train')
y_tst_gdp.plot(color="pink",label='Test')
fcast1.plot(color="purple",label='Forecast')
plt.title("Holt's exponential trend")
plt.legend(loc='best')
plt.show()

print(np.sqrt(mean_squared_error(y_tst_gdp, fcast1)))

###### Autotuning - not mentioning the values of alpha and beta explicitly ###############
######## Exponential = False


holt=Holt(y_trn_gdp)

fit1=holt.fit()
fcast1 =fit1.forecast(len(y_tst_gdp))
  
### plotting
y_trn_gdp.plot(color="blue",label='Train')
y_tst_gdp.plot(color="pink",label='Test')
fcast1.plot(color="purple",label='Forecast')
plt.title("Holt's linear trend")
plt.legend(loc='best')
plt.show()

print(np.sqrt(mean_squared_error(y_tst_gdp, fcast1)))
  

