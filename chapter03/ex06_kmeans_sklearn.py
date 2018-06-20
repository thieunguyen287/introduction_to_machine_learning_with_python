from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, y = make_blobs()

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
predictionns = kmeans.predict(X)
centers = kmeans.cluster_centers_
print 'Cluster memberships:\n', kmeans.labels_
print 'Prediction on training:\n', predictionns
print 'Centers:', centers
plt.scatter(X[:, 0], X[:, 1], c=predictionns, marker='.')
plt.scatter(centers[:, 0], centers[:, 1], c=[0, 1, 2], marker='D', s=50)
plt.show()
