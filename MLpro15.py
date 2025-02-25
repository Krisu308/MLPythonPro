import numpy as np
# Create matrix
matrix_a = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 2]])
# Create matrix
matrix_b = np.array([[1, 3, 1],[1, 3, 1],[1, 3, 8]])
# Add two matrices
print("matrix_a",matrix_a)
print("matrix_b",matrix_b)
print("addition:",np.add(matrix_a, matrix_b))
#array([[ 2, 4, 2], [ 2, 4, 2], [ 2, 4, 10]])
# Subtract two matrices
print("substraction:",np.subtract(matrix_a, matrix_b))
#array([[ 0, -2, 0], [ 0, -2, 0], [ 0, -2, -6]])