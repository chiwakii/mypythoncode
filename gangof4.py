# coding: utf-8
import numpy as np
from control import matlab
from matplotlib import pylab as plt

# 制御対象
num = [1]
den = [1, 1.02, 0.02]
P = matlab.tf(num, den)

# 制御器
num = [50, 1]
den = [50, 0]
C = matlab.tf(num, den)

matlab.gangof4(P, C)
plt.show()
