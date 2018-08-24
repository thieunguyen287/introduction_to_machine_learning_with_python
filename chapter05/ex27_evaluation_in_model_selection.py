from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

X, y = datasets.load_digits(return_X_y=True)
y = y == 9
print "Default scoring: {}".format(cross_val_score(SVC(), X, y))
print "Explicit accuracy scoring: {}".format(cross_val_score(SVC(), X, y, scoring='accuracy'))
print "AUC scoring: {}".format(cross_val_score(SVC(), X, y, scoring='roc_auc'))
