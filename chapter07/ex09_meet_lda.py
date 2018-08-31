import numpy as np
from sklearn import datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import mglearn
import matplotlib.pyplot as plt


reviews_train = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/train')
text_train, y_train = reviews_train.data, reviews_train.target
vectorizer = CountVectorizer(max_features=10000, max_df=0.15)
X_train = vectorizer.fit_transform(text_train)

lda = LatentDirichletAllocation(n_topics=10, learning_method='batch', max_iter=25, random_state=0)
topics = lda.fit_transform(X_train)
print "Topic shape:", topics.shape
print "Component shape:", lda.components_.shape
print "First 5 topics:\n", topics[:5, :]
indices = np.argsort(lda.components_, axis=1)[:, ::-1]
feature_names = np.array(vectorizer.get_feature_names())
mglearn.tools.print_topics(topics=range(10), feature_names=feature_names,
                           sorting=indices, topics_per_chunk=5, n_words=10)
music = np.argsort(topics[:, 5])[::-1]
for i in music:
    print text_train[i]
    print '--' * 30


