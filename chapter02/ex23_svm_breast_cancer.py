from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target)

min_on_training = X_train.min(axis=0)
range_on_training = (X_train - min_on_training).max(axis=0)
scaled_X_train = (X_train - min_on_training) / range_on_training
scaled_X_test = (X_test - min_on_training) / range_on_training

# svc = SVC().fit(X_train, y_train)
svc = SVC(C=1000).fit(scaled_X_train, y_train)

print 'Train accuracy:', svc.score(scaled_X_train, y_train)
print 'Test accuracy:', svc.score(scaled_X_test, y_test)
