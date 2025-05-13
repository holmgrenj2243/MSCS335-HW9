import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#https://www.kaggle.com/datasets/harrywang/wine-dataset-for-clustering
df = pd.read_csv("wine-clustering.csv")
df=df.dropna()

#Pulling data into features
X = df.iloc[:, 1:].to_numpy()

#Normalize
std = np.std(X, axis=0)
X /= std

inertia = np.zeros(14)

#Elbow graph for finding #of clusters
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i).fit(X)
    inertia[i-1] = kmeans.inertia_
plt.plot(np.arange(1, 15), inertia)
plt.show()

# 3 clusters chosen from elbow as well as knowledge from dataset
kmeans = KMeans(n_clusters=3, n_init=20)
kmeans.fit(X)

print(f"Inertia: {kmeans.inertia_}")
for i in range(3):
    print(f"Stats for each cluster {i}: {kmeans.cluster_centers_[i]}")

# Comparison to 4 clusters
kmeans = KMeans(n_clusters=4, n_init=20)
kmeans.fit(X)

print(f"Inertia: {kmeans.inertia_}")
for i in range(4):
    print(f"Stats for each cluster {i}: {kmeans.cluster_centers_[i]}")
