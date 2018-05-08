import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
print 'people images shape:', people.images.shape
print 'Number of classes:', people.target_names
counts = np.bincount(people.target)

# fix, axes = plt.subplots(2, 5, figsize=(15, 8),
#                          subplot_kw={'xticks': (), 'yticks': ()})
# for target, image, ax in zip(people.target, people.images, axes.ravel()):
#     ax.imshow(image)
#     ax.set_title(people.target_names[target])
# plt.show()

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print '{} {}'.format(name, count)
    if (i + 1) % 3 == 0:
        print ''

mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][: 50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people /= 255.

X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
print 'Test score:', knn.score(X_test, y_test)
pca = PCA(n_components=100, whiten=True).fit(X_train)
pca_X_train = pca.transform(X_train)
pca_X_test = pca.transform(X_test)
knn.fit()
print 'PCA test score:', knn
