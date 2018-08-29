import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


print ENGLISH_STOP_WORDS
reviews_train = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/train')
text_train, y_train = reviews_train.data, reviews_train.target
text_train = list(map(lambda s: s.replace(b'<br />', b'\n'), text_train))
print y_train[0], text_train[0]
vectorizer = CountVectorizer(min_df=5, max_df=0.7, stop_words='english')
vectorizer.fit(text_train)

X_train = vectorizer.transform(text_train)
reviews_test = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/test')
text_test, y_test = reviews_test.data, reviews_test.target
text_test = list(map(lambda s: s.replace(b'<b />', b'\n'), text_test))
X_test = vectorizer.transform(text_test)
print np.shape(X_train)

feature_names = vectorizer.get_feature_names()
print "Number of features: {}".format(len(feature_names))
print "First 20 features:\n{}".format(feature_names[:20])
print "Features 20010 to 20030:\n{}".format(feature_names[20010: 20030])
print "Every 2000th feature:\n{}".format(feature_names[::2000])
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print "Mean cross-validation accuracy: {:.2f}".format(np.mean(scores))

param_grid = {'C': [pow(10, p) for p in range(-3, 3)]}
grid = GridSearchCV(LogisticRegression(), param_grid=param_grid, cv=5)
grid.fit(X_train, y_train)
print "Best cross-validation score: {:.2f}".format(grid.best_score_)
print "Best parameters: ", grid.best_params_
print "Test score: {:.2f}".format(grid.score(X_test, y_test))
