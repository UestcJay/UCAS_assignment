#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : QR.py
# @Author: Li 
# @Date  :  2020/12/04

import numpy as np
import math
import os
import sys

from utils import print_mat, load_mat


def QR(mat):
    """ mat,是要分解的矩阵，Gram-Schmidt实现QR分解
    """
    temp = mat.copy()
    m, n = mat.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for col in range(n):
        cur_col = mat[:, col]
        if col == 0:                #a1=||a1||*q1
            R[0, 0] = math.sqrt(np.sum(np.square(cur_col)))
            q = cur_col / R[0, 0]
            Q[:, col] = q
        else:
            q = cur_col.copy()
            for i in range(col):
                R[i, col] = np.matmul(Q[:, i], mat[:, col])
            for j in range(col):        #qi=ai- <q1|ak>q1-<q2|ai>q2- <qi-1|ai>qi-1
                q -= R[j, col] * Q[:, j]
            R[col, col] = math.sqrt(np.sum(np.square(q)))
            q = q / R[col, col]
            Q[:, col] = q
    return Q, R


if __name__ == "__main__":
    path = r'data.txt'
    matrix = load_mat(path, "QR")
    if matrix.size == 0:
        print("input Error!")
        sys.exit()
    Q, R = QR(
        matrix)
    m, n = Q.shape
    print("Q=")
    print_mat(Q, m, m)
    print("R=")
    print_mat(R, m, n)
