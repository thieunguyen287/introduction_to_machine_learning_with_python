from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


X, y = datasets.load_iris(return_X_y=True)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val)

print "Training size: {}, validation size: {}, test size: {}".format(len(X_train), len(X_val), len(X_test))

best_score = 0
best_params = {}
for gamma in [pow(10, p) for p in range(-3, 3)]:
    for C in [pow(10, p) for p in range(-3, 3)]:
        svm = SVC(gamma=gamma, C=C)
        svm.fit(X_train, y_train)
        score = svm.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_params['gamma'] = gamma
            best_params['C'] = C

svm = SVC(**best_params)
svm.fit(X_train_val, y_train_val)
test_score = svm.score(X_test, y_test)
print "Best score on validation set: {:.2f}".format(best_score)
print "Best parameters: ", best_params
print "Test score with best parameters: {:.2f}".format(test_score)
