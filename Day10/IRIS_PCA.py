import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = pd.read_csv(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Datasets-20240429T030248Z-001\Datasets\iris.csv", index_col=0)
iris.head()

scaler = StandardScaler().set_output(transform='pandas')
iris_drop = iris.drop('Species', axis=1)
scaler.fit(iris_drop)

iris_scaled = scaler.transform(iris_drop)

pca = PCA().set_output(transform='pandas')
p_comps = pca.fit_transform(iris_scaled)

###PCA columns are orthogonal to each other (cuz the corr() is very low or 0)
pd.get_dummies().corr()

print(np.cumsum(pca.explained_variance_ratio_*100))

p_comps['Species'] = iris['Species']
sns.scatterplot(data=p_comps, x='pca0', y='pca1', hue='Species')
plt.show()

sns.pairplot(iris, hue = 'Species')
plt.show()