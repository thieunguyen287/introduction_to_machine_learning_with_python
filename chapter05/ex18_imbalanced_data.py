import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

X, y = datasets.load_digits(return_X_y=True)
y = y == 9
X_train, X_test, y_train, y_test = train_test_split(X, y)
dummy = DummyClassifier(strategy='most_frequent')
# dummy = DummyClassifier()
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)
print "Dummy confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred))
print "Unique predicted labels: {}".format(np.unique(y_pred))
print "Dummy test score: {:.2f}".format(dummy.score(X_test, y_test))

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
print "Tree confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred))
print "Tree test score: {:.2f}".format(tree.score(X_test, y_test))

lr = LogisticRegression(C=0.1)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print "LogReg confusion matrix:\n{}".format(confusion_matrix(y_test, y_pred))
print "LogReg test core: {:.2f}".format(lr.score(X_test, y_test))
