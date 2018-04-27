from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)
print X_test.shape
lr = LogisticRegression(C=200, penalty='l1').fit(X_train, y_train)
print 'Train score:', lr.score(X_train, y_train)
print 'Test score:', lr.score(X_test, y_test)

