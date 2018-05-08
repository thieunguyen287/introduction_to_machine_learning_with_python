from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()
scaler = StandardScaler()
scaled_X = scaler.fit_transform(cancer.data)

pca = PCA(n_components=2).fit(scaled_X)

pca_X = pca.transform(scaled_X)

print 'Original shape:', scaled_X.shape
print 'Reduced shape:', pca_X.shape
print 'PCA components:\n', pca.components_

plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(pca_X[:, 0], pca_X[:, 1], cancer.target)
plt.legend(cancer.target_names, loc='best')
plt.gca().set_aspect('equal')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')

# plt.figure()
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0, 1], ['First component', 'Second component'])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names, rotation=60, ha='left')
plt.xlabel('Feature')
plt.ylabel('Principal components')
plt.show()
