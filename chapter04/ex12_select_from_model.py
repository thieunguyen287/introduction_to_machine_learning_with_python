import numpy as np
from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


X, y = datasets.load_breast_cancer(return_X_y=True)
noise = np.random.normal(size=(len(X), 50))
X_noise = np.hstack((X, noise))
print "X_noise.shape:", X_noise.shape
X_train, X_test, y_train, y_test = train_test_split(X_noise, y)
f_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), threshold="median")
# f_selector = SelectFromModel(Ridge(), threshold="median")
f_selector.fit(X_train, y_train)
X_train_selected = f_selector.transform(X_train)
X_test_selected = f_selector.transform(X_test)
print "X_train_selected.shape: {}".format(X_train_selected.shape)
print "X_test_selected.shape: {}".format(X_test_selected.shape)
lr = LogisticRegression().fit(X_train, y_train)
score = lr.score(X_test, y_test)
print "Score on original features: {:.3f}".format(score)

lr = LogisticRegression().fit(X_train_selected, y_train)
score = lr.score(X_test_selected, y_test)
print "Score on selected features: {:.3f}".format(score)

mask = f_selector.get_support()
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks((), ())
plt.show()

