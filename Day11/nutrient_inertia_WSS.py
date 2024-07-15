from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt



df = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\nutrient.csv", index_col=0)
df.info()


scaler = StandardScaler().set_output(transform='pandas')
df_scaled = scaler.fit_transform(df)


clust = KMeans(n_clusters=2, random_state=24)
clust.fit(df_scaled)



print(clust.inertia_)

Ks=[2,3,4,5,6,7,8,9,10]
score=[]
for i in Ks:
    clust= KMeans(n_clusters=i,random_state=24)
    clust.fit(df_scaled)
    score.append(clust.inertia_)
    
plt.scatter(Ks,score,c='red') 
plt.plot(Ks,score)  
plt.title('scree plot')
plt.xlable("Clusters")
plt.ylabel("WSS") 
plt.show()

####################### WSS Method ########


