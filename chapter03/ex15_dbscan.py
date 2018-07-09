from sklearn.cluster import DBSCAN
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples=12)

dbscan = DBSCAN(3, 3)
clusters = dbscan.fit_predict(X)
print 'Cluster memberships:\n', clusters
plt.scatter(X[:, 0], X[:, 1])
plt.show()
