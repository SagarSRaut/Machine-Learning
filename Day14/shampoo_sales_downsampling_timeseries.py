import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import os

os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Day14")

df = pd.read_csv("sales-of-shampoo-over-a-three-ye.csv")
df.head()

df.plot.line(x='Month', y='Sales_shamoo')
plt.show()

y = df['Sales_shamoo']

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
                            damped_trend=True, seasonal='add',use_boxcox=0.5)

# ses= SimpleExpSmoothing(y_train)
fit1=hw_mul.fit(smoothing_level=alpha,smoothing_trend=beta,
                smoothing_seasonal=gamma,damping_trend=phi)
fcast1=fit1.forecast(len(y_test))

y_train.plot(color="blue",label='Train')
y_test.plot(color="green",label='test')
fcast1.plot(color="red",label=' forecast')
error=round(np.sqrt(mean_squared_error(y_test, fcast1)),2)
plt.text(30,200,"RMSE="+str(error))


plt.title("HW multiplicative trend and seasonal method")
plt.legend(loc='best')
plt.show()


#####################################

shamp = pd.read_csv("sales-of-shampoo-over-a-three-ye.csv", index_col=0)
shamp.index = pd.to_datetime(shamp.index).to_period('M')

shamp.plot()
plt.title('Monthly shampoo sales')
plt.show()

shamp_qtr = shamp.resample('Q').sum()
shamp_qtr.index.rename('Quarter', inplace= True)
shamp_qtr.plot()
plt.title('Quarterly shampoo sales')
plt.xlabel('Quarters')
plt.show()



y = shamp_qtr['Sales_shamoo']
y_train = y[:-3]
y_test = y[-3:]

#simple Exponenttail smoothing
from statsmodels.tsa.api import SimpleExpSmoothing,ExponentialSmoothing

alpha=0.8
beta= 0.02
gamma=0.1

hw_add=ExponentialSmoothing(y_train,seasonal_periods=4,trend='add',
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



