import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 101)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('n_neighbors')
# plt.xticks(neighbors_settings, neighbors_settings)
plt.legend()
plt.grid()
plt.figure()
n_neighbors = 3
clf = KNeighborsClassifier(n_neighbors=n_neighbors)
train_sizes, train_scores, test_scores = learning_curve(clf, cancer.data, cancer.target,
                                                        train_sizes=np.linspace(0.1, 1.0, 20),
                                                        cv=KFold(n_splits=20, shuffle=True))
plt.plot(train_sizes, train_scores.mean(axis=1), linestyle='--', label='train score')
plt.plot(train_sizes, test_scores.mean(axis=1), linestyle='-', label='test score')
plt.legend()
plt.grid()
plt.show()
