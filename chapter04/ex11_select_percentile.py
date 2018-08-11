import numpy as np
from sklearn import datasets
from sklearn.feature_selection import SelectPercentile
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()

random_state = np.random.RandomState(0)
noise = random_state.normal(size=(len(cancer.data), 50))
X_noise = np.hstack((cancer.data, noise))
print "X_org.shape:", cancer.data.shape
print "X_noise.shape:", X_noise.shape
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X_noise, y, test_size=.2)
f_selector = SelectPercentile(percentile=50)
f_selector.fit(X_train, y_train)
X_train_selected = f_selector.transform(X_train)
X_test_selected = f_selector.transform(X_test)
print "X_train_selected.shape: {}".format(X_train_selected.shape)
print "X_test_selected.shape: {}".format(X_test_selected.shape)
mask = f_selector.get_support(indices=False)
# mask = f_selector.get_support(indices=True)
print "Selected mask:\n", mask
plt.matshow(mask.reshape(1, -1), cmap='gray_r')
# plt.matshow(np.tile(mask, (2, 1)), cmap='gray_r')
plt.xlabel("Sample index")
plt.yticks((), ())

score = LogisticRegression().fit(X_train, y_train).score(X_test, y_test)
print "Score on original features: {:.3f}".format(score)
score = LogisticRegression().fit(X_train_selected, y_train).score(X_test_selected, y_test)
print "Score on selected features: {:.3f}".format(score)
plt.show()
