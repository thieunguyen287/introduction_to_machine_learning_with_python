import numpy as np
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import mglearn
import matplotlib.pyplot as plt


reviews_train = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/train')
text_train, y_train = reviews_train.data, reviews_train.target
reviews_test = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/test')
text_test, y_test = reviews_test.data, reviews_test.target
pipe = make_pipeline(TfidfVectorizer(min_df=5),
                     LogisticRegression())
param_grid = {'logisticregression__C': [pow(10, p) for p in range(-3, 3)],
              'tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)]}

grid = GridSearchCV(pipe, param_grid, cv=5, verbose=2, n_jobs=3)
grid.fit(text_train, y_train)
print "Best cross-validation score: {:.2f}".format(grid.best_score_)
print "Best parameters: {}\n".format(grid.best_params_)
print "Test score: {:.2f}".format(grid.score(text_test, y_test))
scores = grid.cv_results_['mean_test_score'].reshape(-1, 3).T
vectorizer = grid.best_estimator_.named_steps['tfidfvectorizer']
feature_names = vectorizer.get_feature_names()
coef = grid.best_estimator_.named_steps['logisticregression'].coef_
mglearn.tools.visualize_coefficients(coef, feature_names, n_top_features=40)
plt.figure()
mask = np.array(map(lambda feature_name: len(feature_name.split(' ')) == 3, feature_names))
mglearn.tools.visualize_coefficients(coef.ravel()[mask],
                                     feature_names[mask], n_top_features=40)
heatmap = mglearn.tools.heatmap(scores, xlabel='C', ylabel='ngram_range',
                                cmap='viridis', fmt="%.3f",
                                xticklabels=param_grid['logisticregression__C'],
                                yticklabels=param_grid['tfidfvectorizer__ngram_range'])
plt.colorbar(heatmap)
plt.show()
