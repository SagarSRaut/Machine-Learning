import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns 

milk = pd.read_csv("milk.csv",index_col=0)
milk.head()

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(milk)
milkscaled=scaler.transform(milk)


clust = AgglomerativeClustering(n_clusters=3)
clust.fit(milkscaled)

print(clust.labels_)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(milkscaled)

print(pca.explained_variance_ratio_ * 100) 

principalComponents['Clust'] = clust.labels_
principalComponents['Clust'] = principalComponents['Clust'].astype(str)

sns.scatterplot(data=principalComponents,
                x='pca0', y='pca1', hue='Clust')
for i in np.arange(0, milk.shape[0] ):
    plt.text(principalComponents.values[i,0], 
             principalComponents.values[i,1], 
             list(milk.index)[i])
plt.show()

####### Nutrient
nut = pd.read_csv("nutrient.csv",index_col=0)
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(nut)
nutscaled=scaler.transform(nut)


clust = AgglomerativeClustering(n_clusters=4)
clust.fit(nutscaled)

print(clust.labels_)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(nutscaled)

print(pca.explained_variance_ratio_ * 100) 

principalComponents['Clust'] = clust.labels_
principalComponents['Clust'] = principalComponents['Clust'].astype(str)

sns.scatterplot(data=principalComponents,
                x='pca0', y='pca1', hue='Clust')
for i in np.arange(0, nut.shape[0] ):
    plt.text(principalComponents.values[i,0], 
             principalComponents.values[i,1], 
             list(nut.index)[i])
plt.show()

################### Country Data
country = pd.read_csv(r"C:\Training\Kaggle\Datasets\Country Data - Unsupervised\Country-data.csv",
                      index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(country)
countryscaled=scaler.transform(country)


clust = AgglomerativeClustering(n_clusters=2)
clust.fit(countryscaled)

print(clust.labels_)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(countryscaled)

print(pca.explained_variance_ratio_ * 100) 

principalComponents['Clust'] = clust.labels_
principalComponents['Clust'] = principalComponents['Clust'].astype(str)

sns.scatterplot(data=principalComponents,
                x='pca0', y='pca1', hue='Clust')
for i in np.arange(0, country.shape[0] ):
    plt.text(principalComponents.values[i,0], 
             principalComponents.values[i,1], 
             list(country.index)[i])
plt.show()

df_clusters = pd.DataFrame({'Country':list(country.index),
                            'Labels':clust.labels_})
