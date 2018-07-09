from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = datasets.make_moons(n_samples=200, noise=0.05)

scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

dbscan = DBSCAN()
clusters = dbscan.fit_predict(scaled_X)

plt.scatter(scaled_X[:, 0], X[:, 1], c=clusters)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.show()
