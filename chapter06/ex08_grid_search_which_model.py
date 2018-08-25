from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)
pipe = Pipeline([('preprocessing', StandardScaler()), ('classifier', DummyClassifier())])
param_grid = [{'classifier': [SVC()], 'preprocessing': [StandardScaler(), None],
               'classifier__gamma': [pow(10, p) for p in range(-3, 3)],
               'classifier__C': [pow(10, p) for p in range(-3, 3)]},
              {'classifier': [RandomForestClassifier()], 'preprocessing': [None],
               'classifier__n_estimators': [10, 50, 100],
               'classifier__max_features': [1, 2, 3]}]
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print "Best params:\n{}\n".format(grid.best_params_)
print "Best cross-validation score: {:.2f}".format(grid.best_score_)
print "Test-set scire: {:.2f}".format(grid.score(X_test, y_test))
