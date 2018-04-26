import mglearn
from sklearn.datasets import load_boston
boston = load_boston()
print 'Data shape:', boston.data.shape
X, y = mglearn.datasets.load_extended_boston()
print 'Extended data shape:', X.shape
