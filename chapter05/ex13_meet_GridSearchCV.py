import numpy as np
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from prettytable import PrettyTable
import mglearn
import matplotlib.pyplot as plt

X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
param_grid = {'C': [pow(10, p) for p in range(-3, 3)],
              'gamma': [pow(10, p) for p in range(-3, 3)]}
print "Parameter grid:\n{}".format(param_grid)
grid_search = GridSearchCV(SVC(), param_grid, cv=5, refit=True, return_train_score=True)
grid_search.fit(X_train, y_train)
print "Test score: {:.2f}".format(grid_search.score(X_test, y_test))
print "Best parameters: {}".format(grid_search.best_params_)
print "Best cross-validation score: {:.2f}".format(grid_search.best_score_)
print "Best estimator: {}".format(grid_search.best_estimator_)
results = pd.DataFrame(grid_search.cv_results_)

table = PrettyTable(field_names=list(results.columns))
for r in results.values:
    table.add_row(r)
print table

scores = np.array(results.mean_test_score).reshape(6, 6)
mglearn.tools.heatmap(scores, xlabel='gamma', xticklabels=param_grid['gamma'],
                      ylabel='C', yticklabels=param_grid['C'], cmap="viridis")
plt.show()
