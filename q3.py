import numpy as np

# Get matrix dimensions from user
rows_A = int(input("Enter number of rows for Matrix A: "))
cols_A = int(input("Enter number of columns for Matrix A: "))
rows_B = int(input("Enter number of rows for Matrix B: "))
cols_B = int(input("Enter number of columns for Matrix B: "))

# Validate matrix multiplication condition
if cols_A != rows_B:
    print("Matrix multiplication is not possible due to incompatible dimensions!")
    exit()

# Initialize matrices
matrix_A = []
matrix_B = []

# Read values for Matrix A
print("Enter values for Matrix A:")
for i in range(rows_A):
    row = [int(input()) for _ in range(cols_A)]
    matrix_A.append(row)

# Read values for Matrix B
print("Enter values for Matrix B:")
for i in range(rows_B):
    row = [int(input()) for _ in range(cols_B)]
    matrix_B.append(row)

# Perform manual matrix multiplication
result_matrix = [[0] * cols_B for _ in range(rows_A)]

for i in range(rows_A):
    for j in range(cols_B):
        for k in range(cols_A):
            result_matrix[i][j] += matrix_A[i][k] * matrix_B[k][j]

# Display results
print("\nResult using Manual Multiplication:")
for row in result_matrix:
    print(row)

print("\nResult using NumPy Multiplication:")
print(np.dot(matrix_A, matrix_B))
