import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mglearn

X, y = make_circles(noise=0.25, factor=0.5)

named_y = np.array(['blue', 'red'])[y]
X_train, X_test, y_train, y_test, named_y_train, named_y_test = \
    train_test_split(X, y, named_y)
gbc = GradientBoostingClassifier().fit(X_train, named_y_train)
print gbc.predict(X_test)
print 'X_test shape:', X_test.shape
print 'Decision function shape:', gbc.decision_function(X_test).shape
print 'Decision function:', gbc.decision_function(X_test)
print 'Probabilities:', gbc.predict_proba(X_test)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
mglearn.tools.plot_2d_separator(gbc, X, ax=axes[0], alpha=0.4,
                                fill=True, cm=mglearn.cm2)
scores_images_df = mglearn.tools.plot_2d_scores(gbc, X, ax=axes[1], alpha=.4, cm=mglearn.ReBl)
scores_images_pp = mglearn.tools.plot_2d_scores(gbc, X, ax=axes[2], alpha=.4, cm=mglearn.ReBl, function='predict_proba')

for ax in axes:
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test, markers='^', ax=ax)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, markers='o', ax=ax)
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
cbar = plt.colorbar(scores_images_df, ax=axes.tolist())
axes[0].legend(['Test class 0', 'Test class 1', 'Train class 0', 'Train class 1'], ncol=4, loc=(.1, 1.1))
plt.show()
