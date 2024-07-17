import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Competition_Kaggle\bike-sharing-demand\train.csv", parse_dates=['datetime'])
df.set_index('datetime', inplace=True)

casual = df['casual']
mon_cas = casual.resample('M').sum()
mon_cas.index.rename('Month', inplace=True)
mon_cas.plot()
plt.title('Monthly register')
plt.xlabel('Months')
plt.show()


casual = df['casual']
mon_cas = casual.resample('Q').sum()
mon_cas.index.rename('Quarterly', inplace=True)
mon_cas.plot()
plt.title('Quarterly register')
plt.xlabel('Quarters')
plt.show()



#summing the quarterly results
downsampled = df.resample('QE').sum()
downsampled[['casual','registered']].index.rename('Quarter', inplace=True)
downsampled[['casual','registered']].plot()
plt.title('Quarterly rentals')
plt.xlabel('Quarters')
plt.show()


#summing the monthly results
downsampled_monthly = df.resample('M').sum()
downsampled_monthly[['casual','registered']].index.rename('Month', inplace=True)
downsampled_monthly[['casual','registered']].plot()
plt.title('Monthly rentals')
plt.xlabel('Months')
plt.show()
