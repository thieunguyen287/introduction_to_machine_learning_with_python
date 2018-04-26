import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mglearn.datasets import make_wave
from mglearn.plot_helpers import cm2


X, y = make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

line = np.linspace(-3, 3, 100).reshape(-1, 1)

lr = LinearRegression().fit(X_train, y_train)
print "w: %r  b: %f" % (lr.coef_, lr.intercept_)
print 'Training score:', lr.score(X_train, y_train)
print 'Test score:', lr.score(X_test, y_test)

plt.figure(figsize=(8, 8))
plt.plot(line, lr.predict(line))
plt.plot(X, y, 'o', c=cm2(0))
ax = plt.gca()
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('center')
ax.spines['top'].set_color('none')
ax.set_ylim(-3, 3)
#ax.set_xlabel("Feature")
#ax.set_ylabel("Target")
ax.legend(["model", "training data"], loc="best")
ax.grid(True)
ax.set_aspect('equal')
plt.show()
