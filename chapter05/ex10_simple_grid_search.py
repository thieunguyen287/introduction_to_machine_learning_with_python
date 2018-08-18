import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print "Training size: {}, test size: {}".format(X_train.shape[0], X_test.shape[0])

best_score = -1

gammas = [pow(10, n) for n in range(-3, 3)]
Cs = [pow(10, n) for n in range(-3, 3)]
best_parameters = {}
for gamma in gammas:
    for C in Cs:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
print "Best score: {:.2f}".format(best_score)
print "Best parameters: {}".format(best_parameters)
