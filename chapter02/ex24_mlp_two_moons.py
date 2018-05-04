from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import mglearn
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=100, noise=0.25)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
# mlp = MLPClassifier(hidden_layer_sizes=(10, 10), solver='lbfgs', activation='tanh',
#                     alpha=1.,
#                     max_iter=500, batch_size=32,
#                     early_stopping=True).fit(X_train, y_train)
# print 'Train score:', mlp.score(X_train, y_train)
# print 'Test score:', mlp.score(X_test, y_test)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel('Feature 0')
# plt.ylabel('Feature 1')

fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for axx, n_hidden_nodes in zip(axes, [10, 100]):
    for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1.0]):
        mlp = MLPClassifier(solver='lbfgs',
                            hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes],
                            alpha=alpha)
        mlp.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3, ax=ax)
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        score = mlp.score(X_test, y_test)
        ax.set_title('n_hidden=[{}, {}]\nalpha={}, score={}'.
                     format(n_hidden_nodes, n_hidden_nodes, alpha, score))
plt.show()
