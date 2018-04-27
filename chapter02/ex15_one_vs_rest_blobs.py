import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC

import mglearn
import matplotlib.pyplot as plt

X, y = make_blobs()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
linear_svm = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
print 'Coefficient shape:', linear_svm.coef_.shape
print 'Intercept shape:', linear_svm.intercept_.shape

line = np.linspace(np.min(X), np.max(X))
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# plt.ylim(-10, 15)
# plt.xlim(-10, 8)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'])
plt.show()
