import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("monthly-beer-production-in-austr.csv",
                 index_col=0)
df.index = pd.to_datetime( df.index ).to_period("M")

df.plot()
plt.title("Monthly Beer Production")
plt.xlabel("Months")
plt.show()

downsampled  = df.resample('Q').sum() 
downsampled.index.rename('Quarter', inplace=True)
downsampled.plot()
plt.title("Quarterly Beer Production")
plt.xlabel("Quarters")
plt.show()


############################################################
df = pd.read_csv(r"C:\Training\Kaggle\Competitions\Bike Sharing Demand\train.csv",
                 parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

casual = df['casual']
mon_cas  = casual.resample('M').sum() 
mon_cas.index.rename('Month', inplace=True)
mon_cas.plot()
plt.title("Monthly Casual Rentals")
plt.xlabel("Months")
plt.show()

mon_cas  = casual.resample('Q').sum() 
mon_cas.index.rename('Quarter', inplace=True)
mon_cas.plot()
plt.title("Quarterly Casual Rentals")
plt.xlabel("Quarters")
plt.show()


casual = df['registered']
casual.plot()
plt.show()

mon_cas  = casual.resample('M').sum() 
mon_cas.index.rename('Month', inplace=True)
mon_cas.plot()
plt.title("Monthly Registered Rentals")
plt.xlabel("Months")
plt.show()

mon_cas  = casual.resample('Q').sum() 
mon_cas.index.rename('Quarter', inplace=True)
mon_cas.plot()
plt.title("Quarterly Casual Rentals")
plt.xlabel("Quarters")
plt.show()


##############################################################
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("monthly-milk-production-pounds-p.csv")

series = df['Milk']
result = seasonal_decompose(series, model='additive',
                            period=12)
result.plot()
plt.title("Additive Decomposition")
plt.show()

result = seasonal_decompose(series, model='multiplicative',
                            period=12)
result.plot()
plt.title("Multiplicative Decomposition")
plt.show()




