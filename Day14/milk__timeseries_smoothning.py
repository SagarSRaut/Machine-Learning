import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\monthly-milk-production-pounds-p.csv")
df.head()

df.plot.line(x='Month', y='Milk')
plt.show()

y = df['Milk']

span=3
fcast = y.rolling(span,center=True).mean()
plt.plot(y,label='Data')
plt.plot(fcast,label='Centered rolling mean')
plt.legend(loc='best')
plt.show()


#############################
y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
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

##################################################

# print(sqrt(mean_squared_error(y_test, MA_series))
y_train = df['Milk'][:-12]
y_test = df['Milk'][-12:]
# span = 3
alpha=0.1

#simple  Exponenttail smoothing
from statsmodels.tsa.api import SimpleExpSmoothing
ses= SimpleExpSmoothing(y_train)
fit1=ses.fit(smoothing_level=alpha)
fcast1=fit1.forecast(len(y_test))

 
y_train.plot(color="blue",label='Train'
             )
y_test.plot(color="green",label='test')
fcast1.plot(color="red",label=' forecast')
plt.legend(loc='best')
plt.show()


print(fit1.params)

print('RMSE',np.sqrt(mean_squared_error(y_test, MA_series)))
