import mglearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import euclidean_distances

from mglearn.datasets import make_wave
from mglearn.plot_helpers import cm3
from sklearn.model_selection import train_test_split

X, y = make_wave(n_samples=40)
X_test = np.array([[-1.5], [0.9], [1.5]])

dist = euclidean_distances(X, X_test)
closest = np.argsort(dist, axis=0)

n_neighbors = 1
reg = KNeighborsRegressor(n_neighbors=n_neighbors).fit(X, y)
# plt.figure(figsize=(10, 6))
# y_pred = reg.predict(X_test)
#
# for x, y_, neighbors in zip(X_test, y_pred, closest.T):
#     for neighbor in neighbors[:n_neighbors]:
#             plt.arrow(x[0], y_, X[neighbor, 0] - x[0], y[neighbor] - y_,
#                       head_width=0, fc='k', ec='k')
#
# train, = plt.plot(X, y, 'o', c=cm3(0))
# test, = plt.plot(X_test, -3 * np.ones(len(X_test)), '*', c=cm3(2),
#                  markersize=20)
# pred, = plt.plot(X_test, y_pred, '*', c=cm3(0), markersize=20)
# plt.vlines(X_test, -3.1, 3.1, linestyle="--")
# plt.legend([train, test, pred],
#            ["training data/target", "test data", "test prediction"],
#            ncol=3, loc=(.1, 1.025))
# plt.ylim(-3.1, 3.1)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y)
reg.fit(X_train, y_train)
test_predictions = reg.predict(X_test)
print 'Test set predictions:\n', test_predictions
print 'Test R^2:', reg.score(X_test, y_test)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line), label='Model predictions')
    ax.plot(X_train, y_train, '^', markersize=8, label='Train data/target')
    ax.plot(X_test, y_test, '^', markersize=8, label='Test data/target')
    ax.set_title('{} neighbor(s)\ntrain score: {:.2f}, test score: {:.2f}'.format(
        n_neighbors,
        reg.score(X_train, y_train),
        reg.score(X_test, y_test)))
    ax.set_xlabel('Feature')
    ax.set_ylabel('Target')
    ax.grid()
axes[0].legend(loc='best')
plt.show()
