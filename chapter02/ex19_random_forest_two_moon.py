from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

X, y = make_moons(n_samples=100, noise=0.25)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

forest = RandomForestClassifier(n_estimators=5).fit(X_train, y_train)
# print forest.estimators_
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title('Tree {}'.format(i))
    mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1])
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
print 'Train accuracy:', forest.score(X_train, y_train)
print 'Test accuracy:', forest.score(X_test, y_test)
plt.show()
