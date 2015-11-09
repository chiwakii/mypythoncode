# coding: utf-8
import numpy as np
import scipy as sp
import control as ct

A = np.array([[0, 1],
              [0, -1]])
B = np.array([[0],
              [1]])
Q = np.diag([2, 1])
R = np.array([[1]])

S1 = sp.linalg.solve_continuous_are(A, B, Q, R)
K1 = np.linalg.inv(R).dot(B.T).dot(S1)
E1 = np.linalg.eigvals(A-B.dot(K1))
print("S1 =\n", S1)
print("K1 =\n", K1)
print("E1 =\n", E1)

S2, E2, K2 = ct.care(A, B, Q, R)
print("\nS2 =\n", S2)
print("K2 =\n", K2)
print("E2 =\n", E2)

K3, S3, E3 = ct.matlab.lqr(A, B, Q, R)
print("\nS3 =\n", S3)
print("K3 =\n", K3)
print("E3 =\n", E3)
