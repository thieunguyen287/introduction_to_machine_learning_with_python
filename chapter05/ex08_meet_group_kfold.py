from sklearn import datasets
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


X, y = datasets.make_blobs(n_samples=12, random_state=0)

groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
scores = cross_val_score(LogisticRegression(), X, y, groups, cv=GroupKFold())
print "Cross-validation scores: {}".format(scores)
