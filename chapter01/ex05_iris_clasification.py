from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
print type(iris)
print 'iris keys:', iris.keys()
# print iris['DESCR']
print 'iris target names:', iris['target_names']
print 'Type of data:', type(iris['data'])
print 'Shape of data:', iris['data'].shape
print 'First five columns od data:\n', iris['data'][:5]
print 'Type of target:', type(iris['target'])
print 'Shape of target:', iris['target'].shape
print 'Targets:\n', iris['target']

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
