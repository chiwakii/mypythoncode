#coding: utf-8
import numpy as np
from scipy import signal, interpolate
from matplotlib import pylab as plt

# サンプルデータ作成
t = np.linspace(0, 10, 11)
tt = np.linspace(0, 10, 51)
y = np.sin(t)

# オブジェクト指向型の FITPACK のラッパー
spl1 = interpolate.UnivariateSpline(t, y, s=0)
y1 = spl1(tt)

# UnivariateSpline で s=0 とした場合と同じ
spl2 = interpolate.InterpolatedUnivariateSpline(t, y)
y2 = spl2(tt)

# 非オブジェクト指向型の FITPACK のラッパー
c1 = interpolate.splrep(t, y)
y3 = interpolate.splev(tt, c1)

# 3 次スプライン曲線
c2 = signal.cspline1d(y)
y4 = signal.cspline1d_eval(c2, tt)

# 2 次スプライン曲線
c3 = signal.qspline1d(y)
y5 = signal.qspline1d_eval(c3, tt)

plt.figure()
plt.plot(t, y, "o")
plt.plot(tt, y1)
plt.plot(tt, y2)
plt.plot(tt, y3)
plt.plot(tt, y4)
plt.plot(tt, y5)
plt.show()
