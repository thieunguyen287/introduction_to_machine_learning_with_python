from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


X, y = datasets.load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

pipe = Pipeline([('scaler', MinMaxScaler()), ('svm', SVC())])
param_grid = {'svm__gamma': [pow(10, p) for p in range(-3, 3)],
              'svm__C': [pow(10, p) for p in range(-3, 3)]}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print "Best cross-validation accuracy: {:.2f}".format(grid.best_score_)
print "Test set score: {:.2f}".format(grid.score(X_test, y_test))
print "Best parameters: {}".format(grid.best_params_)
