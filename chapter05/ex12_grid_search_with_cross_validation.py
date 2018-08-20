import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from mglearn import plots
import matplotlib.pyplot as plt

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

best_score = -1
best_params = {}

for gamma in [pow(10, p) for p in range(-3, 3)]:
    for C in [pow(10, p) for p in range(-3, 3)]:
        svm = SVC(C=C, gamma=gamma)
        scores = cross_val_score(svm, X=X_train, y=y_train, cv=5)
        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_params['gamma'] = gamma
            best_params['C'] = C

svm = SVC(**best_params)
svm.fit(X_train, y_train)
test_score = svm.score(X_test, y_test)
print "Best score: {:.2f}".format(best_score)
print "Best params:", best_params
print "Test score: {:.2f}".format(test_score)

# plots.plot_cross_val_selection()
# plt.show()
