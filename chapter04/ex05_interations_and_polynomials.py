import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from mglearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

bins = np.linspace(-3, 3, 11)
print "bins: {}".format(bins)

binned_X = np.digitize(X, bins=bins)
print binned_X
# print np.digitize(line, bins=bins)
encoder = OneHotEncoder(sparse=False).fit(binned_X)
encoded_X = encoder.transform(binned_X)
combined_X = np.hstack([X * encoded_X])

encoded_line = encoder.transform(np.digitize(line, bins=bins))
combined_line = np.hstack([line * encoded_line])

reg = DecisionTreeRegressor(min_samples_split=3).fit(combined_X, y)
plt.plot(line, reg.predict(combined_line), label='decision tree')

reg = LinearRegression().fit(combined_X, y)
plt.plot(line, reg.predict(combined_line), 'g.--', label='linear regression', alpha=0.2)

plt.plot(X[:, 0], y, 'ko')
plt.xlabel('Input feature')
plt.ylabel('Regressionn output')
plt.legend()
plt.show()
