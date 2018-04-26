import numpy as np
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print 'Caner keys:', cancer.keys()
print 'Data shape:', cancer.data.shape
print 'Sample counts per class:\n', {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
print 'Feature names:', cancer.feature_names
