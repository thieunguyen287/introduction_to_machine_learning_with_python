import numpy as np
from mglearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


X, y = datasets.make_blobs(n_samples=(4000, 500), centers=2, cluster_std=(7.0, 2), random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y)

svc = SVC(gamma=0.1)
svc.fit(X_train, y_train)
fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
auc = roc_auc_score(y_test, svc.decision_function(X_test))
print "AUC: {:.3f}".format(auc)
plt.plot(fpr, tpr, label="ROC curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
default_threshold_index = np.argmin(np.abs(thresholds))
plt.plot(fpr[default_threshold_index], tpr[default_threshold_index],
         marker='o', markersize=10, c='k', label='default threshold', fillstyle='none', mew=2)
plt.legend()
plt.show()
