import numpy as np
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from prettytable import PrettyTable
import matplotlib.pyplot as plt


X, y = datasets.load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), Ridge())
param_grid = {'polynomialfeatures__degree': [1, 2, 3],
              'ridge__alpha': [pow(10, p) for p in range(-3, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print "Best param:", grid.best_params_
print "Best score:", grid.best_score_
print grid.cv_results_.keys()

print grid.cv_results_['mean_test_score'].shape

table = PrettyTable(field_names=grid.cv_results_.keys())
print np.shape(grid.cv_results_.values())
for row in np.transpose(grid.cv_results_.values()):
    table.add_row(row)
print table
print "Test score: {:.2f}".format(grid.score(X_test, y_test))
plt.matshow(grid.cv_results_['mean_test_score'].reshape(3, -1),
            vmin=0, cmap='viridis')
plt.xlabel('ridge__alpha')
plt.ylabel('polynomialfeatures_degree')
plt.xticks(range(len(param_grid['ridge__alpha'])), param_grid['ridge__alpha'])
# plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), param_grid['polynomialfeatures__degree'])
plt.yticks(range(len(param_grid['polynomialfeatures__degree'])), [1, 2, 3])
plt.colorbar()
plt.show()
