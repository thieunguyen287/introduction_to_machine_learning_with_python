from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target)
print X_train.shape
print X_test.shape

scaler = MinMaxScaler()
scaler.fit(X_train)

scaled_X_train = scaler.transform(X_train)
print 'transformed shape:', scaled_X_train.shape
print 'per-feature minimum before scaling:\n', X_train.min(axis=0)
print 'per-feature minimum after scaling:\n', scaled_X_train.min(axis=0)
print 'per-feature maximum before scaling:\n', X_train.max(axis=0)
print 'per-feature maximum after scaling:\n', scaled_X_train.max(axis=0)

scaled_X_test = scaler.transform(X_test)
print 'per-feature minimum after scaling:\n', scaled_X_test.min(axis=0)
print 'per-feature maximum after scaling:\n', scaled_X_test.max(axis=0)

svm = SVC(C=100).fit(X_train, y_train)
print 'Test accuracy:', svm.score(X_test, y_test)

svm.fit(scaled_X_train, y_train)
print 'Test accuracy with scaling:', svm.score(scaled_X_test, y_test)
