import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_circles(noise=0.25, factor=0.5)

named_y = np.array(['blue', 'red'])[y]
X_train, X_test, y_train, y_test, named_y_train, named_y_test = \
    train_test_split(X, y, named_y)
gbc = GradientBoostingClassifier().fit(X_train, named_y_train)
print gbc.predict(X_test)
print 'X_test shape:', X_test.shape
print 'Decision function shape:', gbc.decision_function(X_test).shape
print 'Decision function:', gbc.decision_function(X_test)
print 'Probabilities:', gbc.predict_proba(X_test)
