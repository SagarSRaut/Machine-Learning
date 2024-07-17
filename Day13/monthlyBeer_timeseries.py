import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\monthly-beer-production-in-austr.csv", index_col=0)
df.index = pd.to_datetime(df.index).to_period("M")

df.plot()
plt.title('Monthly beer production')
plt.xlabel('Months')
plt.show()


#summing the quarterly results
downsampled = df.resample('Q').sum()
downsampled.index.rename('Quarter', inplace=True)
downsampled.plot()
plt.title('Quarterly beer production')
plt.xlabel('Quarters')
plt.show()

#summing the yearly results
yearly = df.resample('Y').sum()
yearly.index.rename('Yearly', inplace=True)
downsampled.plot()
plt.title('yearly milk production')
plt.xlabel('years')
plt.show()