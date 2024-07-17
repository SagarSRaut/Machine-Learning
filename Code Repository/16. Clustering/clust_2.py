import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.metrics import silhouette_score
import seaborn as sns 

milk = pd.read_csv("milk.csv",index_col=0)
milk.head()

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(milk)
milkscaled=scaler.transform(milk)


Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24)
    clust.fit(milkscaled)
    scores.append(silhouette_score(milkscaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

clust = KMeans(n_clusters=Ks[i_max], random_state=24)
clust.fit(milkscaled)

clust_data = milk.copy()
clust_data['Clust'] = clust.labels_

print( clust_data.groupby('Clust').mean() )

########## Nutrient ##############

nut = pd.read_csv("nutrient.csv",index_col=0)
scaler = StandardScaler().set_output(transform='pandas')

nutscaled=scaler.fit_transform(nut)

Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24)
    clust.fit(nutscaled)
    scores.append(silhouette_score(nutscaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

clust = KMeans(n_clusters=Ks[i_max], random_state=24)
clust.fit(nutscaled)

clust_data = nut.copy()
clust_data['Clust'] = clust.labels_

print( clust_data.groupby('Clust').mean() )

########## Country ##############

country = pd.read_csv(r"C:\Training\Kaggle\Datasets\Country Data - Unsupervised\Country-data.csv",
                      index_col=0)

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(country)
countryscaled=scaler.transform(country)


Ks = [2,3,4,5]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24,
                   init='random')
    clust.fit(countryscaled)
    scores.append(silhouette_score(countryscaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

clust = KMeans(n_clusters=Ks[i_max], random_state=24)
clust.fit(countryscaled)

clust_data = country.copy()
clust_data['Clust'] = clust.labels_

clust_country = clust_data.groupby('Clust').mean() 
clust_country.to_csv(r"C:\Training\Kaggle\Datasets\Country Data - Unsupervised\clust_country.csv",)


g = sns.FacetGrid(clust_data, col="Clust")
g.map(plt.hist, "child_mort")
plt.show()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.boxplot, "health")
plt.show()


g = sns.FacetGrid(clust_data, col="Clust")
g.map(plt.scatter, "child_mort", "total_fer")
plt.show()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(plt.scatter, "gdpp", "income")
plt.show()

clust_corr = clust_data.groupby('Clust').corr()
clust_corr.to_csv(r"C:\Training\Kaggle\Datasets\Country Data - Unsupervised\clust_corr.csv",)

