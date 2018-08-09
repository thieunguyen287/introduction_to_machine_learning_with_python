import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from mglearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

transformer = PolynomialFeatures(degree=5).fit(X)
poly_X = transformer.transform(X)
print "Polynomial feature names:{}".format(transformer.get_feature_names())
print "poly_X shape: {}".format(poly_X.shape)
print "Entries of poly_X:\n{}".format(poly_X[:5])
poly_line = transformer.transform(line)

# reg = DecisionTreeRegressor(min_samples_split=3).fit(poly_X, y)
# plt.plot(line, reg.predict(poly_line), label='decision tree')

reg = LinearRegression().fit(poly_X, y)
plt.plot(line, reg.predict(poly_line), 'g.--', label='linear regression', alpha=0.2)

plt.plot(X[:, 0], y, 'ko')
plt.xlabel('Input feature')
plt.ylabel('Regressionn output')
plt.legend()
plt.show()
