from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

print "Accuracy: {:.3f}".format(accuracy_score(y_test, y_pred))
print "Confusion matrix: \n{}".format(confusion_matrix(y_test, y_pred))
print classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)])
