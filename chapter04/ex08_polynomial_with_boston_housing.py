from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape

scaler = MinMaxScaler().fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2).fit(scaled_X_train)
poly_X_train = poly.transform(scaled_X_train)
poly_X_test = poly.transform(scaled_X_test)
print "poly_X_train shape: {}".format(poly_X_train.shape)
print "poly_X_test shape: {}".format(poly_X_test.shape)
print "poly feature names:\n {}".format(poly.get_feature_names())

ridge = Ridge().fit(scaled_X_train, y_train)
print "Score without interactions: {:.3f}".format(ridge.score(scaled_X_test, y_test))

ridge = Ridge().fit(poly_X_train, y_train)
print "Score with interactions: {:.3f}".format(ridge.score(poly_X_test, y_test))
