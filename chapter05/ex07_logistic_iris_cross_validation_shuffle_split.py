from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit


X, y = datasets.load_iris(return_X_y=True)
lr = LogisticRegression()
# cv = ShuffleSplit(test_size=.5, train_size=.5)
cv = StratifiedShuffleSplit(test_size=.5, train_size=.5)
scores = cross_val_score(lr, X, y, cv=cv)
print "Cross-validation scores: {}".format(scores)

