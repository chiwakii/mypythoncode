# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from scipy.optimize import curve_fit
import numpy as np
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv

def svd(A):
    U, S, V = np.linalg.svd(A)  # 行列Aを特異値分解
    # 結果を表示
    print("U=\n"+str(U))
    print("S=\n"+str(S))
    print("V=\n"+str(V))


def cholesky(A): 
    L = np.linalg.cholesky(A)      # 行列AをQR分解
    # 結果を表示
    print("A=\n"+str(A))
    print("L=\n"+str(L))
    print("L*L^T=\n"+str(L.dot(L.T)))    
    
def qr(A):
    
    Q, R = np.linalg.qr(A)      # 行列AをQR分解
    # 結果を表示
    print("A=\n"+str(A))
    print("Q=\n"+str(Q))
    print("R=\n"+str(R))
    print("Q*R=\n"+str(Q.dot(R)))
    
def functest():
    # xの値を生成
    x = np.arange(-3.14, 3.14, 0.25)
    # 高さを計算
    sin = np.sin(x) # sin(x)の計算
    cos = np.cos(x) # cos(x)の計算
    tan = np.tan(x) # tan(x)の計算
    exp = np.exp(x) # 指数関数の計算
    log = np.log(x) # 対数関数の計算
    # グラフ表示
    plt.plot(x, sin,"-o",lw=2,alpha=0.7,label="sin(x)")
    plt.plot(x, cos,"-o",lw=2,alpha=0.7,label="cos(x)")
    plt.plot(x, tan,"-o",lw=2,alpha=0.7,label="tan(x)")
    plt.plot(x, exp,"-o",lw=2,alpha=0.7,label="exp(x)")
    plt.plot(x, log,"-o",lw=2,alpha=0.7,label="log(x)")
    fn = "Times New Roman"                      # フォント
    plt.tick_params(labelsize=15)               # 軸目盛のフォントサイズ
    plt.xlabel("$x$", fontsize=30, fontname=fn) # x軸のラベル
    plt.ylabel("$y$", fontsize=30, fontname=fn) # y軸のラベル
    plt.xlim([-4, 4])                           # x軸の範囲
    plt.ylim([-4, 4])                           # y軸の範囲
    plt.grid()                                  # グリッドの表示
    plt.legend(fontsize=13)                     # 凡例の表示
    plt.show()                                  # グラフの描画

def linlsq():#最小二乗法
    X = [1,2,3,4,5]
    Y = [1.1, 2.1, 2.8, 4.3, 5.1]

    A = np.array([X,np.ones(len(X))])
    A = A.T
    a,b = np.linalg.lstsq(A,Y)[0]

    plt.plot(X,Y,"ro")
    plt.plot(X,(a*X+b),"g--")
    plt.grid()
    plt.show()
    
def gradient():#勾配
    x = np.arange(0, 10.1, 0.5)
    y = np.arange(0, 10.1, 0.5)
    (xm, ym) = np.meshgrid(x, y)        # グリッドの作成
    (xq,yq) = (5, 5)                    # 電荷の座標
    r = np.sqrt((xq-xm)**2+(yq-ym)**2)  # 電荷との距離
    k = 9.0*10**9                       # 比例定数k
    q = 1                               # 電荷
    E = (k*q)/r**2                      # 2変数の関数(電場E(x,y)
    (Ey,Ex) = np.gradient(E ,.2, .2)    # Eの勾配を計算
    Ex[Ex>0.5],Ey[Ey>0.5] = 0.5, 0.5    # 勾配の上限下限
    Ex[Ex<-0.5],Ey[Ey<-0.5] = -0.5, -0.5
    # 結果を表示
    plt.quiver(xm,ym,-Ex,-Ey,angles="xy",headwidth=3,scale=20,color="#444444")
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.grid()
    plt.legend()
    plt.show()
    
def calcbase():
    A = np.array([[1.,2.]   # 行列Aの生成
                 ,[3.,4.]])
    print("A=\n" + str(A))
    rankA = np.linalg.matrix_rank(A)    # 行列AのRank(階数)を計算
    print("rank(A) = "+str(rankA))
    
    detA = np.linalg.det(A)     # 行列式の計算
    print("det(A)="+str(detA))
    print ("norm(A)="+str(np.linalg.norm(A)))


    B = A.T                 # 行列Aの転置
    print("transposeofA = \n" + str(B))
    #print(np.linalg.det(B))
    invA = np.linalg.inv(A)             # Aの逆行列
    print( "invA=\n" + str(invA) )      # 計算結果の表示
   # print(np.linalg.det(invA))
    #L = np.linalg.cholesky(A)
    #print("L=\n"+str(L))
    
    mean = np.mean(A)              # 算術平均を計算
    print(u"算術平均："+str(mean))    # 結果を表示
    
    y = [1., 2., 3., 4., 5., 5.5, 7., 9.]
    diff_y = np.diff(y)                     # yの差分を計算
    print("y="+str(y))                    # 結果を表示
    print("diff(y)="+str(diff_y))
    
    #連立方程式を解く
    A = np.array([[1.,3.]    # 行列Aの生成
                 ,[4.,2.]])
    B = np.array([1.,1.])   # 行列Bの生成

    X = np.linalg.solve(A, B)
    # 計算結果の表示
    print( "X=\n" + str(X) )
    print np.linalg.eig(A)
    print spla.gmres(A,B)
    
    
def SOR(A, b, tol):
    # 線形連立方程式を SOR 法で解く
    xOld = np.empty_like(b)
    error = 1e12

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = A - L - D

    Mj = np.dot(np.linalg.inv(D), -(L+U))
    rho_Mj = max(abs(np.linalg.eigvals(Mj)))
    w = 2/(1+np.sqrt(1-rho_Mj**2))

    T = np.linalg.inv(D+w*L)
    Lw = np.dot(T, -w*U+(1-w)*D)
    c = np.dot(T, w*b)

    while error > tol:
        x = np.dot(Lw, xOld) + c
        error = np.linalg.norm(x-xOld)/np.linalg.norm(x)
        xOld = x
    return x


def main():
    A = np.array([[7, -1, 0, 1],
              [-1, 9, -2, 2],
              [0, -2, 8, -3],
              [1, 2, -3, 10]])
    b = np.array([-5, 15, -10, 20])

    x = SOR(A, b, 1e-9)
    print(x)    
    

main()
#functest()
#linlsq()
#gradient()
