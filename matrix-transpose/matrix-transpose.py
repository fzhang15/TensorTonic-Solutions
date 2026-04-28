import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)
    M, N = A.shape[0], A.shape[1]
    A_T = np.zeros((N, M))
    for r in range(M):
        for c in range(N):
            A_T[c][r] = A[r][c]

    return A_T
