import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target)
tree = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
print 'Training accuracy:', tree.score(X_train, y_train)
print 'Testing accuracy:', tree.score(X_test, y_test)
print 'Feature importances:\n', tree.feature_importances_
n_features = cancer.data.shape[1]
plt.bar(range(n_features), tree.feature_importances_)
plt.xticks(np.arange(n_features), cancer.feature_names, rotation='vertical')
export_graphviz(tree, out_file='tree.dot', class_names=cancer.target_names,
                feature_names=cancer.feature_names, impurity=False, filled=True)
with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


train_scores = []
test_scores = []
max_depths = range(1, 40)
for max_depth in max_depths:
    tree = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
    train_score = tree.score(X_train, y_train)
    train_scores.append(train_score)
    test_score = tree.score(X_test, y_test)
    test_scores.append(test_score)
plt.figure()
plt.plot(max_depths, train_scores, label='train score')
plt.plot(max_depths, test_scores, label='test score')
plt.grid()
plt.legend()
plt.show()
