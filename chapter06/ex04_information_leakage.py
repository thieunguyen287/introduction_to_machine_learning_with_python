import numpy as np
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


rnd = np.random.RandomState(seed=0)
X = rnd.normal(size=(100, 10000))
y = rnd.normal(size=100)

selector = SelectPercentile(score_func=f_regression, percentile=5)
selector.fit(X, y)
X_selected = selector.transform(X)
print "X_selected.shape: {}".format(X_selected.shape)
print "Cross-validation acc: {:.2f}".format(np.mean(cross_val_score(Ridge(), X_selected, y, cv=5)))

pipe = Pipeline([('select', SelectPercentile(score_func=f_regression, percentile=5)),
                 ('ridge', Ridge())])
print "Cross-validation acc with pipeline: {:.2f}".format(np.mean(cross_val_score(pipe, X, y, cv=5)))
