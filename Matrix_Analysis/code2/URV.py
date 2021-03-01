#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : URV.py
# @Author: Li 
# @Date  :  2020/12/08
import numpy as np
import sys,os
import math

from utils import print_mat, load_mat
from Householder import  Householder_Reduction
from Givens import Givens_Reduction

def URV(mat):
    """ URV分解
    original matrix A为mxn，秩为r, A=URV
    输出U为mxm正交矩阵，R为nxn正交矩阵，R为分块矩阵mxn
    """
    Q,R=Givens_Reduction(mat)
    # print(R)
    r=np.linalg.matrix_rank(mat)
    mat1=R[0:r,:]
    Q1,R1=Givens_Reduction(mat1.T)
    # Q1, R1 = Givens_Reduction(R.T)
    U=Q
    V=Q1
    R1=np.dot(np.dot(U.T,mat),V)
    return U,R1,V

if __name__ == "__main__":
    path=r'data.txt'
    matrix= load_mat(path,"HR")
    if matrix.size ==0:
        print("input Error!")
        sys.exit()
    U,R,V= URV(matrix)
    m, n = U.shape
    m1, n1 = R.shape
    m2, n2 = V.shape
    print("U=")
    print_mat(U, m, n)
    print("R=")
    print_mat(R, m1, n1)
    print("V=")
    print_mat(V, m2, n2)
    print(np.dot(np.dot(U,R),V.T))