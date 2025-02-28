from sklearn.cluster import KMeans 
from skimage import io 
import numpy as np 
# Load image 
image = io.imread('Image.jpg') 
X = image.reshape((-1, 3))  # Reshape to RGB pixels 
# Apply K-means 
kmeans = KMeans(n_clusters=8) 
kmeans.fit(X) 
compressed = kmeans.cluster_centers_[kmeans.labels_] 
compressed_image = compressed.reshape(image.shape) 
# Display 
io.imshow(compressed_image.astype(np.uint8)) 
io.show()
