#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @File  : main.py
# @Author: Li 
# @Date  :  2020/12/04
import numpy as np
import argparse
import utils
import os
import sys
from LU import LU_factorization
from Givens import Givens_Reduction
from Householder import Householder_Reduction
from QR import QR
from URV import URV
from utils import print_mat, load_mat


def main(args):
    path = args.files
    matrix = load_mat(path, args.mode)
    if matrix.size == 0:
        print("input Matrix Error!")
        sys.exit()
    m, n = matrix.shape
    if args.mode == "LU":
        print("LU Factorization, the input should be a square matrix.\n")
    elif args.mode == "QR":
        r = np.linalg.matrix_rank(matrix)
        if r < n:
            print("Error!\n QR Factorization, The matrix has linearly dependent columns can not be uniquely factored as A=QR!\n")
    print("=" * 50, "\norigin matrix type: {m} * {n}".format(m=m, n=n), "\nOrigin Matrix A = ")
    print_mat(matrix, m, n)
    print("\nThe factorization is processing!\n ")
    if args.mode == "LU":
        P, L, U = LU_factorization(matrix)
        m, n = P.shape
        print("L=")
        print_mat(L, m, m)
        print("U=")
        print_mat(U, m, m)
        print("P=")
        print_mat(P, m, m)
    elif args.mode == "QR":
        Q, R = QR(matrix)
        m, n = Q.shape
        m1, n1 = R.shape
        print("Q=")
        print_mat(Q, m, n)
        print("R=")
        print_mat(R, m1, n1)
    elif args.mode == "Householder":
        Q, R = Householder_Reduction(matrix)
        m, n = Q.shape
        m1, n1 = R.shape
        print("Q=")
        print_mat(Q, m, n)
        print("R=")
        print_mat(R, m1, n1)
    elif args.mode == "Givens":
        Q, R = Givens_Reduction(matrix)
        m, n = Q.shape
        m1, n1 = R.shape
        print("Q=")
        print_mat(Q, m, n)
        print("R=")
        print_mat(R, m1, n1)
    elif args.mode == "URV":
        U, R, V = URV(matrix)
        m, n = U.shape
        m1, n1 = R.shape
        m2, n2 = V.shape
        print("U=")
        print_mat(U, m, n)
        print("R=")
        print_mat(R, m1, n1)
        print("V=")
        print_mat(V, m2, n2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Matrix Factorization pipeline')
    parser.add_argument('-f', '--files', default='data.txt', type=str,
                        help='input example file path.')
    parser.add_argument('-m', '--mode', choices=['LU', 'QR', 'Householder', 'Givens', 'URV'], default='QR', type=str,
                        help="['LU','QR','Householder','Givens','URV'], LU->LU factorization, QR->Gram-Schmidt, Householder->Householder Reduction, GR->Givens Reduction, URV-> URV factorization!")
    args = parser.parse_args()
    main(args)
