# coding: utf-8
import numpy as np
import scipy as sp
import control as ct
from matplotlib import pylab as plt


#A = np.array([[0, 1],
#              [0, -1]])
#B = np.array([[0],
#              [1]])
Ap = np.array([[    0,        1,        0,        0],
          [0,    -10.2,        0,  0.01162],
          [0,        0,        0,        1],
          [0,   0.8419,        0,    -10.2]])
 
Bp = np.array([[    0,         0],
      [0.01052,   0.01071],
      [     0,         0],
      [0.08953,  -0.09111 ]])
 
Cp = np.array([[ 1,   0,   0,   0],
      [0,   0,   0,   0],
      [0,   0,   1,   0],
      [0,   0,   0,   0]])
 
Dp = np.array([[  0,   0,],
      [0,   0],
      [0,   0],
      [0,   0]])
Q = np.eye(4,4)
R = np.eye(2,2)

#S1 = sp.linalg.solve_continuous_are(A, B, Q, R)
#K1 = np.linalg.inv(R).dot(B.T).dot(S1)
#E1 = np.linalg.eigvals(A-B.dot(K1))
#print("S1 =\n", S1)
#print("K1 =\n", K1)
#print("E1 =\n", E1)

S2, E2, K2 = ct.care(Ap, Bp, Q, R)
#print("\nS2 =\n", S2)
#print("K2 =\n", K2)
#print("E2 =\n", E2)

print "ctrb = "+ str(np.linalg.matrix_rank(ct.ctrb(Ap,Bp)))
print "obsv = " + str(np.linalg.matrix_rank(ct.obsv(Ap,Cp)))
#K3, S3, E3 = ct.matlab.lqr(Ap, Bp, Q, R)
#print("\nS3 =\n", S3)
#print("K3 =\n", K3)
#print("E3 =\n", E3)
ss_P = ct.ss(Ap,Bp,Cp,Dp)
print ss_P
t = np.linspace(0, 3, 1000)
stepy = ct.step(ss_P,t)
#plt.plot(t,stepy)
#ct.impulse(ss_P)
ct.nyquist(ss_P)
#bode plot only implemented for SISO system.omg
#ct.bode(ss_P)
