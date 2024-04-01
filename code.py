import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
#read data
data=pd.read_csv("C:\\Users\\Admin\\Desktop\\Data_Mining\\findata (1).csv")
data.head()
data.shape
#plot scatter
df=pd.DataFrame(data) #whole data into dataframe
plt.scatter(data['fiscal deficit'],data['capital expenditure'])
plt.xlabel('fiscal deficit')
plt.ylabel('capital expenditure')
#normalize data
dat=data.loc[:,["fiscal deficit","capital expenditure"]]
data_scaled=dat
data_scaled=pd.DataFrame(data_scaled,columns=dat.columns)
data_scaled.head()
#perform k means
#k_mean=KMeans(n_clusters=3)
#k_mean.fit(data_scaled)
distortions = []
K = range(1,10)
for k in K:
kmeanModel = KMeans(n_clusters=k)

kmeanModel.fit(data_scaled)
distortions.append(kmeanModel.inertia_)
print("")
#plotting elbow method for opptimize k
print("Plotting a scatter graph.")
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
print("Using Elbow Method to find the value of k.")
#finally running k means clustering for optimize value of k=4
kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(data_scaled)
identified_clusters = kmeanModel.fit_predict(data_scaled)
identified_clusters
print("")
print("Clustering the states on the basis of fiscal data.")
df['cluster_group']=identified_clusters
print(df)
#plotting clusters
print("")
print("Plotting clusters")
sns.scatterplot(data=data_scaled,x='fiscal deficit',y='capital expenditure',hue=kmeanModel.labels_)
plt.scatter(kmeanModel.cluster_centers_[:,0],kmeanModel.cluster_centers_[:,1],marker="x",c="r",s=80,la
bel="centriods")
plt.legend()
plt.show()
print(" ")

print("Visualizing the clusters with box plot.")
#visualizing the clusters with box plot
fig=plt.figure(figsize=(14,10))
ax1=fig.add_subplot(2,2,1)
ax1=sns.boxplot(x="cluster_group",y="fiscal deficit",data=df)
ax2=fig.add_subplot(2,2,2)
ax2=sns.boxplot(x="cluster_group",y="capital expenditure",data=df)
