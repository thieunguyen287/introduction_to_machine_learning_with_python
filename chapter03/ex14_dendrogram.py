from scipy.cluster import hierarchy
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_blobs(n_samples=12)
linkage_array = hierarchy.ward(X)
hierarchy.dendrogram(linkage_array)
plt.ylabel('Cluster distance')
plt.show()
