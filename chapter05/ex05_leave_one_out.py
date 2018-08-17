from sklearn import datasets
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X, y = datasets.load_iris(return_X_y=True)
loo = LeaveOneOut()
scores = cross_val_score(LogisticRegression(), X, y, cv=loo)
print "Number of cv iterations:", len(scores)
print "Mean accuracy: {:.2f}".format(scores.mean())
