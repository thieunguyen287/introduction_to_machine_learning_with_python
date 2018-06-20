from sklearn import datasets
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# X, y = datasets.make_blobs(random_state=170, n_samples=600)
X, y = datasets.make_blobs(n_samples=600)

# transform the date to be stretched
# rng = np.random.RandomState(74)
# transformation = rng.normal(size=(2, 2))
# transformation = np.random.randn(2, 2)
transformation = np.array([[0.6, -0.6],
                           [-0.4, 0.8]])
print transformation
X = np.dot(X, transformation)

# cluster the data into three clusters
kmeans = KMeans(n_clusters=3).fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=100, c=[1, 2, 3], marker='D')
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
