from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target)

svc = SVC().fit(X_train, y_train)

print 'Train accuracy:', svc.score(X_train, y_train)
print 'Test accuracy:', svc.score(X_test, y_test)
