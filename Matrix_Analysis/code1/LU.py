"""
环境为Python 3.7, 已在win10 Pycharm上运行通过，实验直接从同目录下的data.txt读取需要分解的矩阵A，在终端输出 L,U,P
测试数据为作业题矩阵A:
[1 2 4 17
3 6 -12 3
2 3 -3 2
0 2 -2 6]
输出为：
L=
       1.0,       0.0,       0.0,       0.0,
       0.0,       1.0,       0.0,       0.0,
    0.3333,       0.0,       1.0,       0.0,
    0.6666,      -0.5,       0.5,       1.0
U=
       3.0,       6.0,     -12.0,       3.0,
       0.0,       2.0,      -2.0,       6.0,
       0.0,       0.0,       8.0,      16.0,
       0.0,       0.0,       0.0,      -5.0
P=
       0.0,       1.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       1.0,
       1.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       1.0,       0.0
"""


import numpy as np
import math
import os
import sys


def print_mat(mat, m, n):
    for i in range(m):
        for j in range(n):
            print("%10.6s" % mat[i, j], end=',')
        print()
    print()


def load_mat(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        mat = []
        i = 1
        for line in lines:
            line = line.strip().split(' ')
            line = list(map(eval, line))
            line.append(i)
            i=i+1
            mat.append(line)
        f.close()
        mat=np.array(mat,dtype='float64')
        m,n=mat.shape
        if m!=(n-1):
            print("The shape of matrix is wrong!")
        return mat
def LU_process(mat):
    temp=mat.copy()
    m,n=mat.shape
    for row in range(m-1):
        cur_col=np.abs(mat[row:,row])
        max_i=np.max(cur_col)
        if cur_col[0]!=max_i:
            max_idx=np.where(cur_col==max_i)[0][0] +row
            mat[row]=temp[max_idx]
            mat[max_idx]=temp[row]
        # print_mat(mat, m, n)
        for i in range(row+1,m):
            factor=mat[i,row]/mat[row,row]
            mat[i,row]=factor
            plus=-1*factor*mat[row,row+1:-1]
            mat[i,row+1:-1]+=plus
        temp=mat.copy()
        # print_mat(mat,m,n)
    U = np.triu(mat[:, :-1], 0)
    for i in range(m):
        temp[i,i]=1
    L = np.tril(temp[:, :-1], 0)
    P = np.zeros(shape=(m, n - 1))
    for j in range(m):
        row_idx = int(mat[j, -1])
        P[j, row_idx - 1] = 1
    print("L=")
    print_mat(L,m,m)
    print("U=")
    print_mat(U,m,m)
    print("P=")
    print_mat(P,m,m)
if __name__=="__main__":
    path = os.getcwd()
    testdata= path+"\data.txt"
    mat=load_mat(testdata)
    if mat.size!=0:
        LU_process(mat)




