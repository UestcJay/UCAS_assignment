#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : Givens.py
# @Author: Li 
# @Date  :  2020/12/04
import numpy as np
import math
import os
import sys

from utils import print_mat, load_mat


def Givens_Reduction(mat):
    """ Givens分解
    original matrix A为mxn，A=QR
    Q为mxm正交矩阵，R为mxn上三角矩阵
    """
    m, n = mat.shape
    R = np.copy(mat)
    Q = np.identity(m)
    for i in range(min(m, n)):
        cur_col = R[i + 1:, i]
        if (len(cur_col) == 0):
            break
        for j in range(len(cur_col)):
            if cur_col[j] == 0.:
                continue
            norm = math.sqrt(R[i, i] ** 2 + cur_col[j] ** 2)
            P_cnt = np.identity(m)
            P_cnt[i, i] = R[i, i] / norm                #计算旋转矩阵P_cnt
            P_cnt[i, i + j + 1] = cur_col[j] / norm
            P_cnt[i + j + 1, i] = -cur_col[j] / norm
            P_cnt[i + j + 1, i + j + 1] = R[i, i] / norm

            Q = np.dot(P_cnt, Q)    #Q=P1P2P3...Pn
            R = np.dot(P_cnt, R)    #R=P1P2..PnA
            # Q = np.round(np.dot(P_cnt, Q), 4)
            # R = np.round(np.dot(P_cnt, R), 4)
    return np.round(Q.T, 3), np.round(R, 3)  #保留三位
    # return Q.T,R


if __name__ == "__main__":
    path = r'data.txt'
    matrix = load_mat(path, "HR")
    if matrix.size == 0:
        print("input Error!")
        sys.exit()
    Q, R = Givens_Reduction(matrix)
    a = np.round(np.dot(Q, R), 2)
    print(a)
    m1, _ = Q.shape
    m, n = R.shape
    print("Q=")
    print_mat(Q, m1, m1)
    print("R=")
    print_mat(R, m, n)
