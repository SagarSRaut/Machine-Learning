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

clust = KMeans(n_clusters=2, random_state=24)
clust.fit(milkscaled)

Ks = [2,3,4,5,6,7,8,9,10]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24)
    clust.fit(milkscaled)
    scores.append(clust.inertia_)

plt.scatter(Ks, scores, c='red')
plt.plot(Ks, scores)
plt.title("Scree Plot")
plt.xlabel("Clusters")
plt.ylabel("WSS")
plt.show()