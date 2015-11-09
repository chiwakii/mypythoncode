# -*- coding: utf-8 -*-
from pylab import *
# -10 から 10 まで 0.1 刻みの配列をつくる (numpy.arange )
x  = range(-10.0, 10.0, 0.1)
# 関数 numpy.sin : x の各要素に Math.sin を適用して配列オブジェクトを生成
y = sin(x)
# x,y を描画
plot(x,y)
# 描画
show()
