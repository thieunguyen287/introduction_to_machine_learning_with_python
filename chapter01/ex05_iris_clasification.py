import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from matplotlib.colors import ListedColormap
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

cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
X_train_data_frame = pd.DataFrame(X_train, columns=iris.feature_names)
# grr = pd.plotting.scatter_matrix(X_train_data_frame, c=y_train, figsize=(15, 15), marker='o',
#                                  hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=cm3)
# plt.show()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print 'X_new shape:', X_new.shape
prediction = knn.predict(X_new)
print 'Prediction:', prediction
print 'Predicted target name:', iris['target_names'][prediction]

y_pred = knn.predict(X_test)
print 'Test set predictions:\n', y_pred
print 'Test set:\n', y_test

print 'Test score: {:.2f}'.format(np.mean(y_pred == y_test))
print 'Test score: {:.2f}'.format(knn.score(X_test, y_test))
