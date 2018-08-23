import numpy as np
# from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mglearn import datasets
import matplotlib.pyplot as plt


X, y = datasets.make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# svc = SVC(gamma=.05)
# model = LogisticRegression()
model = RandomForestClassifier(n_estimators=1000)
model.fit(X_train, y_train)
# precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
precision, recall, thresholds = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
# default_index = np.argmin(np.abs(thresholds))
default_index = np.argmin(np.abs(thresholds - 0.5))
plt.plot(precision[default_index], recall[default_index], 'o', markersize=10,
         label='default operating point', fillstyle='none', c='k', mew=2)
plt.plot(precision, recall, label="precision call curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend()
plt.show()
