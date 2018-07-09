from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# X, y = datasets.make_blobs()
X, y = datasets.make_moons()
# X, y = datasets.load_iris(return_X_y=True)
agg = AgglomerativeClustering(n_clusters=3)
predictions = agg.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=predictions)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()
