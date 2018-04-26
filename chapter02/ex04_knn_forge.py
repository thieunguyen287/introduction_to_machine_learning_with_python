import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

from mglearn.datasets import make_forge
from mglearn.plot_helpers import discrete_scatter

X, y = make_forge()

X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])
dist = euclidean_distances(X, X_test)
closest = np.argsort(dist, axis=0)

n_neighbors = 3
clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
k_neighbors = clf.kneighbors(X_test, n_neighbors=n_neighbors, return_distance=False)
for x, neighbor_indices in zip(X_test, k_neighbors):
    for neighbor_index in neighbor_indices:
        plt.arrow(x[0], x[1], X[neighbor_index, 0] - x[0], X[neighbor_index, 1] - x[1],
                  head_width=0, fc='k', ec='k')

test_points = discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers="*")
training_points = discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(training_points + test_points, ["training class 0", "training class 1",
                                           "test pred 0", "test pred 1"])
plt.show()
