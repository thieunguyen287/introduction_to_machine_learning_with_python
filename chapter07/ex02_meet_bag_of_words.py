from sklearn.feature_extraction.text import CountVectorizer


bards_words = ["The fool doth think he is wise",
               "but the be wise man knows himself to be a fool"]

vect = CountVectorizer()
vect.fit(bards_words)
print "Vocabulary size: {}".format(len(vect.vocabulary_))
print "Vocabulary content: {}".format(vect.vocabulary_)
bag_of_words = vect.transform(bards_words)
print "bag_of_words: {}".format(bag_of_words.toarray())