from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics.scorer import SCORERS


X, y = datasets.load_digits(return_X_y=True)
y = y == 9
X_train, X_test, y_train, y_test = train_test_split(X, y)
param_grid = {'gamma': [0.0001, 0.01, 0.1, 1, 10]}


def test_grid_search(metric):
    grid_search = GridSearchCV(SVC(), param_grid, scoring=metric)
    grid_search.fit(X_train, y_train)
    print "Grid-search with {}".format(metric)
    print "Best parameters:", grid_search.best_params_
    print "Best cross-validation score (accuracy): {:.3f}".format(grid_search.best_score_)
    print "Test set auc: {:.3f}".format(roc_auc_score(y_test, grid_search.decision_function(X_test)))
    print "Test set acc: {:.3f}".format(grid_search.score(X_test, y_test))

test_grid_search(metric='accuracy')
test_grid_search(metric='roc_auc')
print SCORERS.keys()
