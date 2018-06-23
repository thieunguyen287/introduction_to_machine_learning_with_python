from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# generate synthetic two_moons data (with less noise this time)
X, y = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)

# cluster the data into two clusters
kmeans = KMeans(n_clusters=2).fit(X)
y_pred = kmeans.predict(X)

# plot the cluster assignment and cluster centers
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c=[0, 1], s=100)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

