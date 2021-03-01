import numpy as np
import math
import os
import sys
from utils import print_mat, load_mat


def LU_factorization(mat):
    """ PLU分解，mat为要分解的矩阵，输出P,L,U3个矩阵
    """
    temp = mat.copy()
    m, n = mat.shape
    for row in range(m - 1):
        cur_col = np.abs(mat[row:, row])
        max_i = np.max(cur_col)
        if cur_col[0] != max_i:
            max_idx = np.where(cur_col == max_i)[0][0] + row
            mat[row] = temp[max_idx]
            mat[max_idx] = temp[row]
        # print_mat(mat, m, n)
        for i in range(row + 1, m):
            factor = mat[i, row] / mat[row, row]
            mat[i, row] = factor
            plus = -1 * factor * mat[row, row + 1:-1]
            mat[i, row + 1:-1] += plus
        temp = mat.copy()
        # print_mat(mat,m,n)
    U = np.triu(mat[:, :-1], 0)
    for i in range(m):
        temp[i, i] = 1
    L = np.tril(temp[:, :-1], 0)
    P = np.zeros(shape=(m, n - 1))
    for j in range(m):
        row_idx = int(mat[j, -1])
        P[j, row_idx - 1] = 1
    return P, L, U


if __name__ == "__main__":
    path = r"data.txt"
    matrix = load_mat(path, "LU")
    if matrix.size == 0:
        print("input Error!")
        sys.exit()
    P, L, U = LU_factorization(matrix)
    m, n = P.shape
    print("L=")
    print_mat(L, m, m)
    print("U=")
    print_mat(U, m, m)
    print("P=")
    print_mat(P, m, m)
