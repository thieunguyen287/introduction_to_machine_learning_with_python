import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier

from mglearn.datasets import make_forge
from mglearn.plot_helpers import discrete_scatter
from sklearn.model_selection import train_test_split
import mglearn

X, y = make_forge()

X_test = np.array([[8.2, 3.66214339], [9.9, 3.2], [11.2, .5]])
dist = euclidean_distances(X, X_test)
closest = np.argsort(dist, axis=0)

n_neighbors = 3
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
# clf.fit(X, y)
# k_neighbors = clf.kneighbors(X_test, n_neighbors=n_neighbors, return_distance=False)
# for x, neighbor_indices in zip(X_test, k_neighbors):
#     for neighbor_index in neighbor_indices:
#         plt.arrow(x[0], x[1], X[neighbor_index, 0] - x[0], X[neighbor_index, 1] - x[1],
#                   head_width=0, fc='k', ec='k')
#
# test_points = discrete_scatter(X_test[:, 0], X_test[:, 1], clf.predict(X_test), markers="*")
# training_points = discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(training_points + test_points, ["training class 0", "training class 1",
#                                            "test pred 0", "test pred 1"])
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf.fit(X_train, y_train)
test_predictions = clf.predict(X_test)
print 'Test set predictions:', test_predictions
print 'Test accuracy:', clf.score(X_test, y_test)
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    clf.decision_function()
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title('{} neighbor(s)'.format(n_neighbors))
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')

axes[0].legend(loc=3)
plt.show()
