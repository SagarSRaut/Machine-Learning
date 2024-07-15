from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score



df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Cases-20240426T111123Z-001\Cases\Recency Frequency Monetary\rfm_data_customer.csv", index_col=2)
df.drop('customer_id', axis=1, inplace=True)

scaler = StandardScaler().set_output(transform='pandas')
df_scaled = scaler.fit_transform(df)


clust = KMeans(n_clusters=2, random_state=24)
clust.fit(df_scaled)

 

print(clust.labels_)


df_clust = df.copy()
df_clust['Clust'] = clust.labels_
df_clust['Clust'] = df_clust['Clust'].astype(str)


Ks = [2,3,4,5,6]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24)
    clust.fit(df_scaled)
    scores.append(silhouette_score(df_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])


clust = KMeans(n_clusters=Ks[i_max], random_state=24)
clust.fit(df_scaled)

clust_data = df.copy()
clust_data['Clust'] = clust.labels_
    

print(clust_data.groupby('Clust').mean())
print(clust_data['Clust'].value_counts())


clust_corr = clust_data.groupby('Clust').corr()


g = sns.FacetGrid(clust_data, col='Clust')
g.map(plt.scatter, 'number_of_orders', 'recency_days')
plt.show()

g = sns.FacetGrid(clust_data, col='Clust')
g.map(sns.histplot, 'recency_days')
plt.show()

g = sns.FacetGrid(clust_data, col='Clust')
g.map(sns.histplot, 'number_of_orders')
plt.show()

g = sns.FacetGrid(clust_data, col='Clust')
g.map(sns.histplot, 'revenue')
plt.show()


plt.scatter(Ks,scores,c='red') 
plt.plot(Ks,scores)  
plt.title('scree plot')
plt.xlabel("Clusters")
plt.ylabel("WSS") 
plt.show()

####################### WSS Method ########


