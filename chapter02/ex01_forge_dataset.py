import matplotlib.pyplot as plt
import mglearn

X, y = mglearn.datasets.make_forge()
print 'X shape:', X.shape
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['Class 0', 'Class 1'], loc=4)
plt.xlabel('First feature')
plt.ylabel('Second feature')
plt.show()
