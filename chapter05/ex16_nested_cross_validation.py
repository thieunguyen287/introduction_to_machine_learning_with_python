from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


X, y = datasets.load_iris(return_X_y=True)
param_grid = {'gamma': [pow(10, p) for p in range(-3, 3)],
              'C': [pow(10, p) for p in range(-3, 3)]}
scores = cross_val_score(GridSearchCV(SVC(), param_grid=param_grid, cv=5), X, y, cv=5)
print "Cross-validation scores:", scores
print "Mean cross-validation score:", scores.mean()
