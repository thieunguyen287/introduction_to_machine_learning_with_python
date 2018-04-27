import numpy as np
from scipy import linalg
from sklearn.linear_model import ElasticNet

import mglearn
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y)
lasso = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print 'Training score:', lasso.score(X_train, y_train)
print 'Test score:', lasso.score(X_test, y_test)
print 'Number of features used:', np.sum(lasso.coef_ != 0)

alphas = np.linspace(0.001, 1.0, 200)
coef_mags = []
train_scores = []
test_scores = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=int(1000 / alpha)).fit(X_train, y_train)
    coef_mags.append(linalg.norm(lasso.coef_))
    train_scores.append(lasso.score(X_train, y_train))
    test_scores.append(lasso.score(X_test, y_test))

alphas = np.linspace(0.001, 1.0, 200)
elastic_coef_mags = []
elastic_train_scores = []
elastic_test_scores = []
for alpha in alphas:
    elastic = ElasticNet(alpha=alpha).fit(X_train, y_train)
    elastic_coef_mags.append(linalg.norm(elastic.coef_))
    elastic_train_scores.append(elastic.score(X_train, y_train))
    elastic_test_scores.append(elastic.score(X_test, y_test))

plt.plot(alphas, coef_mags, label='lasso')
plt.plot(alphas, elastic_coef_mags, label='elastic')
plt.grid()
plt.legend()
plt.figure()
plt.plot(alphas, train_scores, label='train score')
plt.plot(alphas, elastic_train_scores, label='elastic train score')
plt.plot(alphas, test_scores, label='test score')
plt.plot(alphas, elastic_test_scores, label='elastic test score')
plt.legend()
plt.show()
