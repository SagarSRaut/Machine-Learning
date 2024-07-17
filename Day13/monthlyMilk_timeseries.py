import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\monthly-milk-production-pounds-p.csv", index_col=0)
df.index = pd.to_datetime(df.index).to_period("M")

df.plot()
plt.title('Monthly milk production')
plt.xlabel('Months')
plt.show()


#summing the quarterly results
downsampled = df.resample('Q').mean()
downsampled.index.rename('Quarter', inplace=True)
downsampled.plot()
plt.title('Quarterly milk production')
plt.xlabel('Quarters')
plt.show()

