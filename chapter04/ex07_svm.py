import numpy as np
from mglearn import datasets
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X, y = datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, num=1000, endpoint=False).reshape(-1, 1)

for gamma in [1, 10]:
    svr = SVR(gamma=gamma, C=1000).fit(X, y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X[:, 0], y, 'o', c='k')
plt.xlabel('Input feature')
plt.ylabel('Regression output')
plt.legend(loc='best')
plt.show()
