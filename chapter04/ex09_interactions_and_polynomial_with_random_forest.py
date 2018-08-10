from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = MinMaxScaler().fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)

forest = RandomForestRegressor(n_estimators=100).fit(scaled_X_train, y_train)
print "Forest score without polynomial: {:.3f}".format(forest.score(scaled_X_test, y_test))

poly = PolynomialFeatures().fit(scaled_X_train)
poly_X_train = poly.transform(scaled_X_train)
poly_X_test = poly.transform(scaled_X_test)
forest = RandomForestRegressor(n_estimators=100).fit(poly_X_train, y_train)
print "Forest score with polynomial: {:.3f}".format(forest.score(poly_X_test, y_test))
