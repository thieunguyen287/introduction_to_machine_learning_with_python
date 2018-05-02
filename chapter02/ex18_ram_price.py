import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

ram_prices = pd.read_csv('../data/ram_price.csv')
# plt.semilogy(ram_prices.date, ram_prices.price)
# plt.xlabel('Year')
# plt.ylabel('Price in $/Mbyte')
# plt.show()

train_data = ram_prices[ram_prices.date < 2000]
test_data = ram_prices[ram_prices.date >= 2000]

X_train = train_data.date[:, np.newaxis]
y_train = np.log(train_data.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

X_all = ram_prices.date[:, np.newaxis]

tree_pred = tree.predict(X_all)
lr_pred = linear_reg.predict(X_all)

tree_price = np.exp(tree_pred)
lr_price = np.exp(lr_pred)

plt.semilogy(train_data.date, train_data.price, label='Training data')
plt.semilogy(test_data.date, test_data.price, label='Test data')
plt.semilogy(ram_prices.date, tree_price, label='Tree prediction')
plt.semilogy(ram_prices.date, lr_price, label='Linear prediction')
plt.legend()
plt.show()
