from sklearn.cluster import MeanShift
import numpy as np
import matplotlib.pyplot as plt 
# Data 
X = np.array([[1, 2], [2, 3], [3, 4], [10, 11], [11, 12], [12, 13]]) 
# Apply Mean Shift 
meanshift = MeanShift() 
meanshift.fit(X) 
# Plotting 
plt.scatter(X[:, 0], X[:, 1], c=meanshift.labels_) 
plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], c='red') 
plt.show() 
