import numpy as np
from scipy import sparse

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print 'coo representation:\n{}'.format(eye_coo)
