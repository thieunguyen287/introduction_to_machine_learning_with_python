from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

X, y = datasets.load_iris(return_X_y=True)
lr = LogisticRegression()
kfold = KFold(n_splits=3, shuffle=True)
scores = cross_val_score(lr, X, y, cv=kfold)
print scores
