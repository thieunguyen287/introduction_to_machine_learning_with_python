import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


reviews_train = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/train')
text_train, y_train = reviews_train.data, reviews_train.target
reviews_test = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/test')
text_test, y_test = reviews_test.data, reviews_test.target

pipe = make_pipeline(TfidfVectorizer(min_df=5, norm=False, max_df=0.7),
                     LogisticRegression())
param_grid = {'logisticregression__C': [pow(10, p) for p in range(-3, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(text_train, y_train)
print "Best cross-validation score: {:.2f}".format(grid.best_score_)
print "Test score: {:.2f}".format(grid.score(text_test, y_test))
vectorizer = grid.best_estimator_.named_steps['tfidfvectorizer']
X_train = vectorizer.transform(text_train)
max_values = X_train.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_values.argsort()
feature_names = np.array(vectorizer.get_feature_names())

print "Features with lowest tfid:\n{}".format(feature_names[sorted_by_tfidf[:20]])
print "Features with highest tfid:\n{}".format(feature_names[sorted_by_tfidf[-20:]])
sorted_by_idf = np.argsort(vectorizer.idf_)
print "Features with lowest idf:\n{}".format(feature_names[sorted_by_idf[:100]])
