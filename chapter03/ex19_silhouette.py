import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import silhouette_score
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
scaler = StandardScaler().fit(X)
scaled_X = scaler.transform(X)

fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
axes[0].scatter(scaled_X[:, 0], scaled_X[:, 1], c=random_clusters, s=60)
axes[0].set_title("Random assignment: {:.2f}".format(silhouette_score(scaled_X, random_clusters)))

algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(), DBSCAN()]

for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(scaled_X)
    ax.set_title("{}: {:.2f}".format(algorithm.__class__.__name__, silhouette_score(scaled_X, clusters)))
    ax.scatter(scaled_X[:, 0], scaled_X[:, 1], c=clusters, s=60)

plt.show()
