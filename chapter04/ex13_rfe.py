import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X, y = datasets.load_breast_cancer(return_X_y=True)
print "X.shape: {}, y.shape: {}".format(X.shape, y.shape)
noise = np.random.normal(size=(len(X), 50))
X_noise = np.hstack((X, noise))
print "X_noise.shape: {}".format(X_noise.shape)
X_train, X_test, y_train, y_test = train_test_split(X_noise, y)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

f_selector = RFE(RandomForestClassifier(n_estimators=100), None)
f_selector.fit(X_train_scaled, y_train)
X_train_selected = f_selector.transform(X_train_scaled)
X_test_selected = f_selector.transform(X_test_scaled)

score = LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
print "Score on noise data: {:.3f}".format(score)

score = LogisticRegression().fit(X_train_scaled, y_train).score(X_test_scaled, y_test)
print "Score on scaled data: {:.3f}".format(score)

score = LogisticRegression().fit(X_train_selected, y_train).score(X_test_selected, y_test)
print "Score on data with selected features: {:.3f}".format(score)

mask = f_selector.get_support().reshape(1, -1)
plt.matshow(mask, cmap="gray_r")
plt.xlabel("Sample index")
plt.yticks((), ())
plt.show()
