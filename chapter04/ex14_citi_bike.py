import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

from mglearn import datasets
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def load_citibike(data_path):
    df = pd.read_csv(data_path)
    df['one'] = 1
    df['starttime'] = pd.to_datetime(df.starttime)
    df = df.set_index('starttime')
    print df.resample('3h').one.head()
    return df.resample('3h').sum().fillna(0).one


def evaluate_on_features(features, target, regressor, n_train, xticks):
    X_train, X_test = features[:n_train], features[n_train:]
    y_train, y_test = target[:n_train], target[n_train:]
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(X_train)
    X_train = encoder.transform(X_train)
    X_test = encoder.transform(X_test)
    poly_transformer = PolynomialFeatures(interaction_only=True, include_bias=False)
    X_train = poly_transformer.fit_transform(X_train)
    X_test = poly_transformer.transform(X_test)
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)
    print "Test-set R^2: {:.2f}".format(score)
    y_test_pred = regressor.predict(X_test)
    y_train_pred = regressor.predict(X_train)
    plt.figure(figsize=(10, 3))
    plt.xticks(range(0, len(X), 8), xticks, rotation=90)
    plt.plot(range(n_train), y_train, label='train')
    plt.plot(range(n_train, len(X)), y_test, '-', label='test')
    plt.plot(range(n_train), y_train_pred, '--', label='train prediction')
    plt.plot(range(n_train, len(X)), y_test_pred, '--', label='test prediction')
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    # plt.show()
    plt.figure()
    hour = ["%02d:00" % i for i in range(0, 24, 3)]
    day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    feature_names = day + hour
    poly_features = poly_transformer.get_feature_names(feature_names)
    non_zero_features = np.array(poly_features)[regressor.coef_ != 0]
    non_zero_coef = regressor.coef_[regressor.coef_ != 0]
    plt.plot(non_zero_coef, 'o')
    plt.xticks(range(len(non_zero_coef)), non_zero_features, rotation=90)
    plt.xlabel("Feature magnitude")
    plt.ylabel("Feature")



data_path = '/home/thieunguyen/Code/Python/Learning/introduction_to_machine_learning_with_python/data/citibike.csv'
# citibike = datasets.load_citibike()
citibike = load_citibike(data_path)
print "Citi Bike data:\n{}".format(citibike.head())
print type(citibike)
# plt.figure(figsize=(10, 3))
# xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')
# plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90)
# plt.plot(citibike, linewidth=1)
# plt.xlabel("Date")
# plt.ylabel("Rentals")
# plt.show()

y = citibike.values
# X = citibike.index.strftime('%s').astype('int').reshape(-1, 1)
# X = np.reshape(citibike.index.hour.values, (-1, 1))
X = np.hstack((citibike.index.hour.values.reshape(-1, 1),
               citibike.index.dayofweek.values.reshape(-1, 1)))

data_size = len(X)
test_index = int(0.9 * data_size)
while citibike.index[test_index].strftime('%d') == citibike.index[test_index - 1].strftime('%d'):
    test_index -= 1
print citibike.index[test_index - 1], citibike.index[test_index]
# X_train = X[:test_index]
# X_test = X[test_index:]
# y_train = y[:test_index]
# y_test = y[test_index:]

# regressor = RandomForestRegressor(n_estimators=100)
regressor = LinearRegression()
evaluate_on_features(X, y, regressor, test_index, xticks=pd.date_range(start=citibike.index.min(),
                                                                       end=citibike.index.max(),
                                                                       freq='D'))
plt.show()
