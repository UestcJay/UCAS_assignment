#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : utils.py
# @Author: Li 
# @Date  :  2020/12/04

import numpy as np
import math
import os
import sys


def print_mat(mat, m, n):
    """ 打印输出矩阵
    mat为矩阵，m,n为mat的行和列数
    """
    for i in range(m):
        for j in range(n):
            if j==n-1:
                print("%10.6s" % mat[i, j])
            else:
                print("%10.6s" % mat[i, j], end=',')
        print()
    print()


def load_mat(fname,mode):
    """ 从文件中加载矩阵
    fname为数据文件，mode的选择的模式，输出为一个初始矩阵
    """
    with open(fname, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        mat = []
        i = 1
        for line in lines:
            line = line.strip().split(' ')
            line = list(map(eval, line))
            if mode =='LU':
                line.append(i)
                i=i+1
            mat.append(line)
        f.close()
        mat=np.array(mat,dtype='float64')
        m,n=mat.shape
        if mode =='LU'and m!=(n-1):
            print("The shape of matrix is wrong!")
            sys.exit()
        return mat

