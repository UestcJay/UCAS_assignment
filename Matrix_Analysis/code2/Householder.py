#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : Householder.py
# @Author: Li 
# @Date  :  2020/12/04
import numpy as np
import math
import os
import sys

from utils import print_mat, load_mat


def Householder_Reduction(mat):
    """ Householder分解
    original matrix A为mxn，A=QR
    Q为mxm正交矩阵，R为mxn上三角矩阵
    """
    m, n = mat.shape
    R = np.copy(mat)
    Q = np.identity(m)
    for idx in range(min(m, n)):
        cur_matrix = mat[idx:, idx:]
        if (cur_matrix.shape == (1, 1) or (m < n and cur_matrix.shape == (1, 2))):
            break
        x = R[idx:, idx]
        if np.linalg.norm(x) == 0.:
            continue
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_cnt = np.identity(m)
        Q_cnt[idx:, idx:] -= 2.0 * np.outer(v, v) #
        R = np.dot(Q_cnt, R)  #R=P1P2..PnA
        Q = np.dot(Q_cnt, Q)  #Q=P1P2P3...Pn
    return np.round(Q.T, 3), np.round(R, 3) #保留三位
    # return Q.T,R


if __name__ == "__main__":
    path = r'data.txt'
    matrix = load_mat(path, "HR")
    if matrix.size == 0:
        print("input Error!")
        sys.exit()
    Q, R = Householder_Reduction(matrix)
    m, _ = Q.shape
    m, n = R.shape
    print(np.round(np.dot(Q, R), 2))
    print("Q=")
    print_mat(Q, m, m)
    print("R=")
    print_mat(R, m, n)
