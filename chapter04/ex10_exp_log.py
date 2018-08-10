import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


random_state = np.random.RandomState(seed=0)
X_org = random_state.normal(size=(1000, 3))
w = random_state.normal(size=3)
print np.exp(X_org)
X = random_state.poisson(10 * np.exp(X_org))
y = np.dot(X_org, w)
bins = np.bincount(X[:, 0])
X_train, X_test, y_train, y_test = train_test_split(X, y)
score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print "Test score: {:.3f}".format(score)

X_train_log = np.log(X_train + 1)
X_test_log = np.log(X_test + 1)

score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
print "Test score: {:.3f}".format(score)
# plt.bar(range(len(bins)), bins, color='k')
# plt.xlabel('Value')
# plt.ylabel('Number of apperances')
# plt.show()

