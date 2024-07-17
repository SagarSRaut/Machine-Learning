import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\FMAC-HPI_24420.csv")
df.drop('NSA Value', axis=1, inplace=True)
df.head()

df.plot.line(x='Date', y='SA Value')
plt.show()

y = df['SA Value']

span=3
fcast = y.rolling(span,center=True).mean()
plt.plot(y,label='Data')
plt.plot(fcast,label='Centered rolling mean')
plt.legend(loc='best')
plt.show()


#############################
y_train = y[:-12]
y_test = y[-12:]
# span = 3

fcast = y_train.rolling(span).mean()
MA = fcast.iloc[-1]
MA_series = pd.Series(MA.repeat(len(y_test)))
MA_fcast = pd.concat([fcast, MA_series], ignore_index=True)

plt.plot(y_train,label='train')
plt.plot(y_test,label='test')
plt.plot(MA_fcast,label='Moving average forecast')
plt.legend(loc='best')
plt.show()

rms = np.sqrt(mean_squared_error(y_test, MA_series))

##################### Multiplicative #############################

# print(sqrt(mean_squared_error(y_test, MA_series))
y_train = y[:-12]
y_test = y[-12:]
# span = 3


#simple  Exponenttail smoothing
from statsmodels.tsa.api import SimpleExpSmoothing,ExponentialSmoothing

alpha=0.8
beta= 0.02
gamma=0.1

hw_add=ExponentialSmoothing(y_train,seasonal_periods=12,trend='add',
                             seasonal='add')

# ses= SimpleExpSmoothing(y_train)
fit1=hw_add.fit()
fcast1=fit1.forecast(len(y_test))

y_train.plot(color="blue",label='Train')
y_test.plot(color="green",label='test')
fcast1.plot(color="red",label=' forecast')
error=round(np.sqrt(mean_squared_error(y_test, fcast1)),2)
plt.text(120,90,"RMSE="+str(error))


plt.title("HW additive trend and seasonal method")
plt.legend(loc='best')
plt.show()

############# Seasonal Additive & Damned ####################

y_train = y[:-12]
y_test = y[-12:]
# span = 3

#simple  Exponential smoothing
from statsmodels.tsa.api import SimpleExpSmoothing,ExponentialSmoothing

alpha=0.8
beta= 0.02
gamma=0.1

hw_add=ExponentialSmoothing(y_train,seasonal_periods=12,trend='add',
                            damped_trend=True, seasonal='add')

# ses= SimpleExpSmoothing(y_train)
fit1=hw_add.fit(smoothing_level=alpha,smoothing_trend=beta,
                smoothing_seasonal=gamma)
fcast1=fit1.forecast(len(y_test))

y_train.plot(color="blue",label='Train')
y_test.plot(color="green",label='test')
fcast1.plot(color="red",label=' forecast')
error=round(np.sqrt(mean_squared_error(y_test, fcast1)),2)
plt.text(120,90,"RMSE="+str(error))


plt.title("HW additive trend and seasonal method")
plt.legend(loc='best')
plt.show()


############# Seasonal multiplicative & Damned ####################

y_train = y[:-12]
y_test = y[-12:]
# span = 3

#simple  Exponenttail smoothing
from statsmodels.tsa.api import SimpleExpSmoothing,ExponentialSmoothing

alpha=0.8
beta= 0.02
gamma=0.1
phi=0.1
hw_mul=ExponentialSmoothing(y_train,seasonal_periods=12,trend='add',
                            damped_trend=True, seasonal='add')

# ses= SimpleExpSmoothing(y_train)
fit1=hw_mul.fit(smoothing_level=alpha,smoothing_trend=beta,
                smoothing_seasonal=gamma,damping_trend=phi)
fcast1=fit1.forecast(len(y_test))

y_train.plot(color="blue",label='Train')
y_test.plot(color="green",label='test')
fcast1.plot(color="red",label=' forecast')
error=round(np.sqrt(mean_squared_error(y_test, fcast1)),2)
plt.text(120,90,"RMSE="+str(error))


plt.title("HW additive trend and seasonal method")
plt.legend(loc='best')
plt.show()








