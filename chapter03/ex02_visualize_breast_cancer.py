import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import mglearn

cancer = load_breast_cancer()
fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

axes = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins=50)
    axes[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    axes[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    axes[i].set_title(cancer.feature_names[i])
    axes[i].set_yticks(())

axes[0].set_xlabel('Feature magnitude')
axes[0].set_ylabel('Frequency')
axes[0].legend()
fig.tight_layout()
plt.show()
