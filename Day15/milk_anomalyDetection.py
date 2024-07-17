import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
import seaborn as sns 
import os

os.chdir(r"C:\Users\Administrator.DAI-PC2\Desktop\jupiter_demo\ML\Day15")
df = pd.read_csv("milk.csv", index_col=0)
############################################################################

clf = IsolationForest(contamination=0.04, random_state=24)
clf.fit(df)
predictions = clf.predict(df)

print("%age of outliers="+ str((predictions<0).mean()*100)+ "%")
abn_ind = np.where(predictions < 0)
print("Outliers:")
print(df.index[abn_ind])

###### Visualization of Outliers ################
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
prcomp = PCA()
scores = prcomp.fit_transform(scaled_df)

print(np.cumsum(prcomp.explained_variance_ratio_))

obs = np.where(predictions == -1, "Outlier", "Inliner")
PCs = pd.DataFrame({'PC1':scores[:,0], 'PC2':scores[:,1],
                    'Class':obs})

sns.scatterplot(data=PCs, x='PC1',
                y='PC2', hue='Class')
for i in np.arange(0, df.shape[0] ):
    plt.text(scores[i,0], scores[i,1], 
             list(df.index)[i],fontsize=6)
plt.legend(loc='best')
plt.title("PCA")
plt.show()

