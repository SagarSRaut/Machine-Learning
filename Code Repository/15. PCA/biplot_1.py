import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from numpy import linalg as LA 

milk = pd.read_csv("milk.csv", index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
m_scaled = scaler.fit_transform(milk)

prcomp = PCA().set_output(transform='pandas')
components = prcomp.fit_transform(m_scaled)

print(components.var())
print(prcomp.explained_variance_)
# var_cov = np.cov(m_scaled.T)
# values, vectors = LA.eig(var_cov)
# print(values)

tot_var = np.sum(prcomp.explained_variance_)
print(prcomp.explained_variance_/tot_var)
perc = (prcomp.explained_variance_/tot_var)*100
print(perc)
print(np.cumsum(perc))

#########################################
from pca import pca
import matplotlib.pyplot as plt 

model = pca()
results = model.fit_transform(m_scaled,
          col_labels=milk.columns,
          row_labels=list(milk.index))
model.biplot(label=True,legend=True)
for i in np.arange(0, milk.shape[0] ):
    plt.text(components.values[i,0], 
             components.values[i,1], 
             list(milk.index)[i])
plt.show()











