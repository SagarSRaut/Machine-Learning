import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import seaborn as sns 

milk = pd.read_csv("milk.csv",index_col=0)
milk.head()

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(milk)
milkscaled=scaler.transform(milk)

pca = PCA().set_output(transform='pandas')
principalComponents = pca.fit_transform(milkscaled)
### PCA columns are orthogonal to each other
principalComponents.corr()

print(principalComponents.var())
## Variances of PC columns are eigen values of 
## var-cov matrix
values, vectors = np.linalg.eig(milkscaled.cov())

print(pca.explained_variance_)
tot_var = np.sum(pca.explained_variance_)
print(pca.explained_variance_/tot_var)
# or
print(pca.explained_variance_ratio_) 
print(pca.explained_variance_ratio_ * 100) 

print(np.cumsum(pca.explained_variance_ratio_ * 100)) 


ys = np.cumsum(pca.explained_variance_ratio_ * 100)
xs = np.arange(1,6)
plt.plot(xs,ys)
plt.show()

##################### Iris ####################
iris = pd.read_csv("iris.csv")
sns.pairplot(iris, hue='Species')
plt.show()

iris = pd.read_csv("iris.csv")
iris.head()

scaler = StandardScaler().set_output(transform='pandas')
scaler.fit(iris.drop('Species', axis=1))
iris_scaled=scaler.transform(iris.drop('Species', axis=1))

pca = PCA().set_output(transform='pandas')
p_comps = pca.fit_transform(iris_scaled)

print(np.cumsum(pca.explained_variance_ratio_ * 100)) 

p_comps['Species'] = iris['Species']
sns.scatterplot(data=p_comps, x='pca0', y='pca1',
                hue='Species')
plt.show()
