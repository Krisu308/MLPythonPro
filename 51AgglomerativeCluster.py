from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import AgglomerativeClustering 
from scipy.cluster.hierarchy import dendrogram, linkage 
# Data 
X = np.array([[5, 3], [10, 15], [15, 12], [24, 10], [30, 30], [85, 70], [71, 80], [60, 78], 
[70, 55]]) 
# Apply Agglomerative Clustering 
cluster = AgglomerativeClustering(n_clusters=3) 
labels = cluster.fit_predict(X) 
# Plot Dendrogram 
linked = linkage(X, 'ward') 
dendrogram(linked) 
plt.show() 
