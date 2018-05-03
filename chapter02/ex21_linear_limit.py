from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

X, y = make_blobs(centers=4)
y %= 2
print X.shape
X0 = X[y == 0]
X1 = X[y == 1]
linear_svm = LinearSVC().fit(X, y)
print linear_svm.coef_, linear_svm.intercept_
plt.scatter(X0[:, 0], X0[:, 1], c='b', marker='o', label='class 1')
plt.scatter(X1[:, 0], X1[:, 1], c='r', marker='^', label='class 2')
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
plt.legend()
plt.show()
