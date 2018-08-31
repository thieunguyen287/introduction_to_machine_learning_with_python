import re
import spacy
import nltk
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


regexp = re.compile('(?u)\\b\\w\\w+\\b')

en_nlp = spacy.load('en_core_web_sm')
# old_tokenizer = en_nlp.tokenizer
# en_nlp.tokenizer = lambda s: old_tokenizer.tokens_from_list(regexp.findall(s))


def custom_tokenizer(doc):
    doc_spacy = en_nlp(doc)
    return [token.lemma_ for token in doc_spacy]

lemma_vect = CountVectorizer(tokenizer=custom_tokenizer, min_df=5)

reviews_train = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/train')
text_train, y_train = reviews_train.data, reviews_train.target
# reviews_test = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/test')
# text_test, y_test = reviews_test.data, reviews_test.target

pipe_lemma = make_pipeline(lemma_vect, LogisticRegression())
pipe_normal = make_pipeline(CountVectorizer(min_df=5), LogisticRegression())

param_grid = {'logisticregression__C': [pow(10, p) for p in range(-3, 3)]}
grid_lemma = GridSearchCV(pipe_lemma, param_grid, cv=5, verbose=2, n_jobs=3)
grid_normal = GridSearchCV(pipe_normal, param_grid, cv=5, verbose=2, n_jobs=3)

grid_lemma.fit(text_train, y_train)
print "Best cross-validation score of grid_lemma: {:.3f}".format(grid_lemma.best_score_)
grid_normal.fit(text_train, y_train)
print "Best cross-validation score of grid_normal: {:.3f}".format(grid_normal.best_score_)
