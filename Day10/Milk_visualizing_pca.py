import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

milk = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\milk.csv", index_col=0)
milk.head()

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(milk)

m_scaled = scaler.transform(milk)

prin_comps = PCA().set_output(transform='pandas')
p_comps = prin_comps.fit_transform(m_scaled)

###PCA columns are orthogonal to each other (cuz the corr() is very low or 0)

print(np.cumsum(prin_comps.explained_variance_ratio_*100))


sns.scatterplot(data=p_comps, x='pca0', y='pca1')
plt.show()

####################################################
from pca import pca
model = pca()
results = model.fit_transform(m_scaled, col_labels=milk.columns, 
                              row_labels=list(milk.index))

model.biplot(label=True, legend=True)
for i in np.arange(0, milk.shape[0]):
    plt.text(p_comps.values[i,0], p_comps.values[i,1], list(milk.index)[i])
    
plt.show()