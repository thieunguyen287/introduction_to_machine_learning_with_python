from mglearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

X, y = datasets.make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
apc_rf = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
print "Average precision of random forest: {:.3f}".format(apc_rf)

model = SVC(gamma=0.01)
model.fit(X_train, y_train)
apc_svc = average_precision_score(y_test, model.decision_function(X_test))
print "Average precision of SVC: {:.3f}".format(apc_svc)
