from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    shuffle=True, stratify=iris.target)
bayes = GaussianNB().fit(X_train, y_train)
print 'Train score:', bayes.score(X_train, y_train)
print 'Test score:', bayes.score(X_test, y_test)
