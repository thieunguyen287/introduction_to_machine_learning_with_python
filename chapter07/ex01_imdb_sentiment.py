import numpy as np
from sklearn import datasets


reviews_train = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/train')
text_train, y_train, target_names = reviews_train.data, reviews_train.target, reviews_train.target_names
text_train = map(lambda s: s.replace(b'<br />', b'\n'), text_train)
print "Type of text train: {}".format(type(text_train))
print "Length of text train: {}".format(len(text_train))
print "Target names: {}".format(target_names)
print y_train[0], text_train[0]
print "Samples per class (training): {}".format(np.bincount(y_train))

reviews_test = datasets.load_files('/home/thieunguyen/Datasets/aclImdb/test')
text_test, y_test = reviews_test.data, reviews_test.target
text_test = list(map(lambda s: s.replace(b'<br />', b'\n'), text_test))
print "Number of documents int test data: {}".format(len(text_test))
print "Samples per class (test): {}".format(np.bincount(y_test))
