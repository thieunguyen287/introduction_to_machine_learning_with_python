from scipy import sparse
import numpy as np

eye = np.eye(4)
print 'Numpy eye:\n{}'.format(eye)

sparse_matrix = sparse.csr_matrix(eye)
print 'Scipy sparse csr matrix:\n{}'.format(sparse_matrix)

print sparse_matrix + eye
print sparse_matrix + sparse_matrix
