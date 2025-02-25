# Load libraries
import numpy as np
from scipy import sparse
# Create a matrix
matrix = np.array([[0, 0], [0, 1], [3, 0]])
print("Simple matrix")
print(matrix)
# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)
print("Sparse matrix")
print(matrix_sparse)
