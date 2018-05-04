from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target)
mean_on_train = X_train.mean(axis=0)
std_on_train = X_train.std(axis=0)
scaled_X_train = (X_train - mean_on_train) / std_on_train
scaled_X_test = (X_test - mean_on_train) / std_on_train
mlp = MLPClassifier(max_iter=1000, alpha=1.0).fit(scaled_X_train, y_train)

print 'Train accuracy:', mlp.score(scaled_X_train, y_train)
print 'Test accuracy:', mlp.score(scaled_X_test, y_test)
