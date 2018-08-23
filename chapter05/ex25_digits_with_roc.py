from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


X, y = datasets.load_digits(return_X_y=True)
y = y == 9
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

for gamma in [1.0, 0.05, 0.01]:
    svc = SVC(gamma=gamma)
    svc.fit(X_train, y_train)
    acc = svc.score(X_test, y_test)
    auc = roc_auc_score(y_test, svc.decision_function(X_test))
    print "Gamma: {:.2f}, acc: {:.2f}, auc: {:.2f}".format(gamma, acc, auc)
    fpr, tpr, thresholds = roc_curve(y_test, svc.decision_function(X_test))
    plt.plot(fpr, tpr, label="gamma {:.3f}".format(gamma))
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
