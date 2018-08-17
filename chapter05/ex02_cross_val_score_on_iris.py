from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


X, y = datasets.load_iris(return_X_y=True)
lr = LogisticRegression()
scores = cross_val_score(lr, X, y, cv=3)
print scores
print scores.mean()

