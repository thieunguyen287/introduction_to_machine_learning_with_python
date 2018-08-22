from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


X, y = datasets.load_digits(return_X_y=True)
y = y == 9
X_train, X_test, y_train, y_test = train_test_split(X, y)


def evaluate(estimator, name):
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print "F1 score for {}: {:.2f}".format(name, f1)
    print classification_report(y_test, y_pred, target_names=['0', '1'])

evaluate(DummyClassifier(), 'dummy')
evaluate(DummyClassifier(strategy='most_frequent'), 'most_frequent')
evaluate(DecisionTreeClassifier(), 'tree')
evaluate(LogisticRegression(), 'log reg')
