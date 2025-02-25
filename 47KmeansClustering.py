import numpy as np 
from sklearn.cluster import KMeans 
import matplotlib.pyplot as plt 
# Data 
X = np.array([[10, 15], [20, 30], [15, 25], [50, 55], [60, 70], [65, 75]]) 
# K-means 
kmeans = KMeans(n_clusters=2) 
kmeans.fit(X) 
# Plotting 
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', 
marker='x') 
plt.show() 
