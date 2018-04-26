from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

import mglearn
from sklearn.model_selection import train_test_split

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y)
# lr = LinearRegression().fit(X_train, y_train)
lr = Ridge(alpha=0.1).fit(X_train, y_train)

print 'Train score:', lr.score(X_train, y_train)
print 'Test score:', lr.score(X_test, y_test)

coef_mags = []
alphas = np.arange(0.1, 10.0, 0.1)
for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    coef_mag = linalg.norm(ridge.coef_)
    coef_mags.append(coef_mag)

plt.plot(alphas, coef_mags)
plt.show()
