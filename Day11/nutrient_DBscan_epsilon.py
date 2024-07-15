from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd


nutrient = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\nutrient.csv", index_col=0)
nutrient.info()



from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler().set_output(transform='pandas')
nutrientscaled=scaler.fit_transform(nutrient)

clust_DB = DBSCAN(eps=1, min_samples=2)
clust_DB.fit(nutrientscaled)
print(clust_DB.labels_)

clust_nutrient = nutrient.copy()
clust_nutrient["Clust"] = clust_DB.labels_
clust_nutrient.sort_values(by='Clust')

clust_nutrient.groupby('Clust').mean()
clust_nutrient.sort_values('Clust')


nutrientscaled['Clust'] = clust_DB.labels_
nutrient_scl_inliers = nutrientscaled[nutrientscaled['Clust']!=-1]
print( silhouette_score(nutrient_scl_inliers.iloc[:,:-1],
                 nutrient_scl_inliers.iloc[:,-1]) )

eps_range = [0.2,0.4,0.6,1]
mp_range = [2,3,4,5]
cnt = 0
a =[]
for i in eps_range:
    for j in mp_range:
        clust_DB = DBSCAN(eps=i, min_samples=j)
        clust_DB.fit(nutrientscaled.iloc[:,:5])
        if len(set(clust_DB.labels_)) > 2:
            cnt = cnt + 1
            nutrientscaled['Clust'] = clust_DB.labels_
            nutrient_scl_inliers = nutrientscaled[nutrientscaled['Clust']!=-1]
            sil_sc = silhouette_score(nutrient_scl_inliers.iloc[:,:-1],
                             nutrient_scl_inliers.iloc[:,-1])
            a.append([cnt,i,j,sil_sc])
            print(i,j,sil_sc)
    
a = np.array(a)
pa = pd.DataFrame(a,columns=['Sr','eps','min_pt','sil'])
print("Best Paramters:")
pa[pa['sil'] == pa['sil'].max()]

### Labels with best parameters
clust_DB = DBSCAN(eps=0.4, min_samples=2)
clust_DB.fit(nutrientscaled.iloc[:,:5])
print(clust_DB.labels_)


clust_nutrient = nutrient.copy()
clust_nutrient["Clust"] = clust_DB.labels_
clust_nutrient.sort_values(by='Clust')


clust_nutrient.groupby('Clust').mean()
clust_nutrient.sort_values('Clust')

clust_nutrient["Clust"].value_counts()

