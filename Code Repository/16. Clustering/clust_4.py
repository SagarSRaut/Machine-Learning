import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import seaborn as sns 
import numpy as np 

rfm = pd.read_csv(r"C:\Training\Academy\Statistics (Python)\Cases\Recency Frequency Monetary\rfm_data_customer.csv",index_col=0)
rfm.head()

rfm.drop('most_recent_visit', axis=1, inplace=True)
scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(rfm)
rfm_scaled=scaler.transform(rfm)


Ks = [2,3,4,5,6]
scores = []
for i in Ks:
    clust = KMeans(n_clusters=i, random_state=24)
    clust.fit(rfm_scaled)
    scores.append(silhouette_score(rfm_scaled, clust.labels_))

i_max = np.argmax(scores)
print("Best no. of clusters:", Ks[i_max])
print("Best Score:", scores[i_max])

clust = KMeans(n_clusters=Ks[i_max], random_state=24)
clust.fit(rfm_scaled)

clust_data = rfm.copy()
clust_data['Clust'] = clust.labels_

print( clust_data.groupby('Clust').mean() )
print( clust_data['Clust'].value_counts() )

clust_corr = clust_data.groupby('Clust').corr()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(plt.scatter, "number_of_orders", "revenue")
plt.show()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(plt.scatter, "number_of_orders", "recency_days")
plt.show()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(plt.scatter, "recency_days", "revenue")
plt.show()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.histplot, "recency_days")
plt.show()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.histplot, "number_of_orders")
plt.show()

g = sns.FacetGrid(clust_data, col="Clust")
g.map(sns.histplot, "revenue")
plt.show()

#########################################
eps_range = [0.6,1,1.5, 2]
mp_range = [20, 50, 100]
cnt = 0
a =[]
for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(rfm_scaled.iloc[:,:5])
        if len(set(clust_DB.labels_)) > 2:
            cnt = cnt + 1
            rfm_scaled['Clust'] = clust_DB.labels_
            rfm_scl_inliers = rfm_scaled[rfm_scaled['Clust']!=-1]
            sil_sc = silhouette_score(rfm_scl_inliers.iloc[:,:-1],
                             rfm_scl_inliers.iloc[:,-1])
            a.append([cnt,i,j,sil_sc])
            print(i,j,sil_sc)
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]

### Labels with best parameters
clust_DB = DBSCAN(eps=0.6, min_samples=100)
clust_DB.fit(rfm_scaled.iloc[:,:5])
print(clust_DB.labels_)

clust_rfm = rfm.copy()
clust_rfm["Clust"] = clust_DB.labels_
clust_rfm.sort_values(by='Clust')

clust_rfm.groupby('Clust').mean()
clust_rfm.sort_values('Clust')



