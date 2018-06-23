from sklearn import datasets
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

X, y = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)

# kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
kmeans = MiniBatchKMeans(n_clusters=10, random_state=0).fit(X)
y_pred = kmeans.predict(X)

distance_features = kmeans.transform(X)
print 'Distance features shape:', distance_features.shape
print 'Distance features:', distance_features

# second_kmeans = KMeans(n_clusters=2).fit(distance_features)
# y_pred = second_kmeans.predict(distance_features)

plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Paired')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='^', c=range(kmeans.n_clusters), cmap='Paired', s=60)
plt.xlabel('Feature 0')
plt.ylabel('Feature 1')
print 'Cluster memberships:\n', y_pred
plt.show()
