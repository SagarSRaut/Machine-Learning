import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 

country = pd.read_csv(r"C:\Training\Kaggle\Datasets\Country Data - Unsupervised\Country-data.csv",
                      index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
m_scaled = scaler.fit_transform(country)

prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(m_scaled)
print(prcomp.explained_variance_)

perc = prcomp.explained_variance_ratio_ * 100
print(perc)
print(np.cumsum(perc))

#########################################
from pca import pca
import matplotlib.pyplot as plt 

model = pca()
results = model.fit_transform(m_scaled,
          col_labels=country.columns,
          row_labels=list(country.index))
model.biplot(label=True,legend=True)
for i in np.arange(0, country.shape[0] ):
    plt.text(components.values[i,0], 
             components.values[i,1], 
             list(country.index)[i])
plt.show()
