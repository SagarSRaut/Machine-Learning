import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pylab as plt

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\monthly-milk-production-pounds-p.csv")

series = df['Milk']
result = seasonal_decompose(series, model='additive', period=12)
result.plot()

plt.title('Additive decomposition')
plt.show()


####### centered rolling mean #######

y = df['Milk']

span = 3
fcast = y.rolling(span, center=True).mean()
plt.plot(y, label='Data')
plt.plot(fcast, label='centered rolling mean')
plt.legend(loc='best')
plt.show()

fcast = y.rolling(span, center=True).mean()
plt.plot(fcast, label='centered rolling mean')
plt.legend(loc='best')
plt.show()