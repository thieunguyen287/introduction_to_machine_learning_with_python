import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    stratify=cancer.target)
gbc = GradientBoostingClassifier(learning_rate=0.05).fit(X_train, y_train)

print 'Train accuracy:', gbc.score(X_train, y_train)
print 'Test accuracy:', gbc.score(X_test, y_test)

print 'Cross validation:', np.mean(cross_val_score(gbc, cancer.data, cancer.target, cv=5))
