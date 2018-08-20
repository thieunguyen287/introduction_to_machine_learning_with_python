from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
from prettytable import PrettyTable
param_grid = [{'kernel': ['rbf'],
               'C': [pow(10, p) for p in range(-3, 3)],
               'gamma': [pow(10, p) for p in range(-3, 3)]},
              {'kernel': ['linear'],
               'C': [pow(10, p) for p in range(-3, 3)]}]
print "List of grids:\n{}".format(param_grid)
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
grid_search = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid_search.fit(X_train, y_train)
print "Best parameters: {}".format(grid_search.best_params_)
print "Best cross-validation score: {:.2f}".format(grid_search.best_score_)
results = pd.DataFrame(grid_search.cv_results_)
table = PrettyTable(field_names=list(results.columns))
for r in results.values:
    table.add_row(r)
print table
