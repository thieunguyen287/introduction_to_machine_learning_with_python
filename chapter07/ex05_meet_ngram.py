from sklearn.feature_extraction.text import CountVectorizer


bards_words = ['The fool doth think he is wise',
               'but the wise man knows himself to be a fool',
               'It is good, not bad at all',
               'It is bad, not good at all']

# vectorizer = CountVectorizer(ngram_range=(1, 1))
# vectorizer = CountVectorizer(ngram_range=(2, 2))
vectorizer = CountVectorizer(ngram_range=(1, 3))
vectorizer.fit(bards_words)
print "Vocabulary size: {}".format(len(vectorizer.vocabulary_))
print "Vocabulary: \n{}".format(vectorizer.vocabulary_)
