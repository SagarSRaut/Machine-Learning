import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("monthly-milk-production-pounds-p.csv",
                 index_col=0)
df.index = pd.to_datetime( df.index ).to_period("M")

df.plot()
plt.title("Monthly Milk Production")
plt.xlabel("Months")
plt.show()

downsampled  = df.resample('Q').sum() 
downsampled.index.rename('Quarter', inplace=True)
downsampled.plot()
plt.title("Quarterly Milk Production")
plt.xlabel("Quarters")
plt.show()

#################### Shampoo Sales ########################
shamp = pd.read_csv("sales-of-shampoo-over-a-three-ye.csv",
                    index_col=0)
idx = pd.to_datetime( shamp.index ).to_period("M")
shamp.index = idx

shamp.plot()
plt.title("Monthly Shampoo Sales")
plt.show()

shamp_qtr  = shamp.resample('Q').sum() 
shamp_qtr.index.rename('Quarter', inplace=True)
shamp_qtr.plot()
plt.title("Quarterly Shampoo Sales")
plt.xlabel("Quarters")
plt.show()

##################### Daily Data ##########################
temp_data = pd.read_csv("daily-minimum-temperatures-in-me.csv", 
                        index_col=0)
idx = pd.to_datetime( temp_data.index ).to_period("D")
temp_data.index = idx

temp_data.plot()
plt.title("Daily Min Temperature")
plt.xlabel("Days")
plt.show()


temp_month  = temp_data.resample('M').mean() 
temp_month.index.rename('Monthly', inplace=True)
temp_month.plot()
plt.title("Monthly Avg Min Temperature")
plt.xlabel("Months")
plt.show()


temp_month  = temp_data.resample('M').min() 
temp_month.index.rename('Monthly', inplace=True)
temp_month.plot()
plt.title("Monthly Min Temperature")
plt.xlabel("Months")
plt.show()
