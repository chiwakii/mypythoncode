# -*- coding: utf-8 -*-
"""
Created on Wed Nov 04 00:55:57 2015

@author: chiwakii
"""


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from control import matlab
import control as ctl

DT = 0.05 # Sampling interval - defaults to None, must be set explicitly

# ============================================================
# Components

class Component:
    def work( self, u ):
        return u

    def monitoring( self ):
        return ""  # Overload, to include addtl monitoring info in output

# ============================================================
# Controllers

# --- PID Controllers

class PidController( Component ):
    def __init__( self, kp, ki, kd=0 ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0
        self.d = 0
        self.prev = 0

    def work( self, e ):
        self.i += DT*e
        self.d = ( e - self.prev )/DT
        self.prev = e

        return self.kp*e + self.ki*self.i + self.kd*self.d

class AdvController( Component ):
    def __init__( self, kp, ki, kd=0, clamp=(-1e10,1e10), smooth=1 ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i = 0
        self.d = 0
        self.prev = 0

        self.unclamped = True
        self.clamp_lo, self.clamp_hi = clamp
        
        self.alpha = smooth
    
    def work( self, e ):
        if self.unclamped:
            self.i += DT*e
            
        self.d = self.alpha*(e - self.prev)/DT + (1.0-self.alpha)*self.d

        u = self.kp*e + self.ki*self.i + self.kd*self.d

        self.unclamped = ( self.clamp_lo < u < self.clamp_hi )
        self.prev = e
                
        return u

# --- Relay and Band Controllers

class DeadbandController( Component ):
    def __init__( self, zone ):
        self.zone = zone

    def work( self, e ):
        if e>self.zone:
            return e - self.zone
        elif e<-self.zone:
            return e + self.zone
        else:
            return 0

class RelayController( Component ):
    def work( self, e ):
        if e == 0:
            return 0
        return e/abs(e)

class DeadbandRelayController( Component ):
    def __init__( self, zone ):
        self.zone = zone

    def work( self, e ):
        if e>self.zone:
            return 1
        elif e<-self.zone:
            return -1
        else:
            return 0

class HysteresisRelayController( Component ):
    def __init__( self, zone ):
        self.zone = zone
        self.prev = None

    def work( self, e ):
        
        if e > self.prev: # raising
            if e < self.zone:
                u = 0
            else:
                u = 1
        else:             # falling
            if e > -self.zone:
                u = 0
            else:
                u = -1

        self.prev = e
        return u

# ============================================================
# Simple Systems : Plant

class Boiler( Component ):
    # Default g: temp drops to 1/e in 100 secs (for water: approx 1 deg/sec)
    # Work u: input is change in temp (per sec), if no heat loss
    
    def __init__( self, g=0.01 ):
        self.y = 0      # initial state, "temperature" (above ambient)
        self.g = g      # constant of proportionality (time constant)

    def work( self, u ):
        self.y += DT*( -self.g*self.y + u )
        return self.y

class Spring( Component ):
    # In mks units (defaults: 100g, 1N/m, approx 10 periods to fall to 1/e)
    
    def __init__( self, m=0.1, k=1, g=0.05 ):
        self.x = 0      # position
        self.v = 0      # velocity

        self.m = m      # mass
        self.k = k      # spring constant: Newton/meter
        self.g = g      # damping factor

    def work( self, u ):
        a = ( - self.k*self.x - self.g*self.v + u )/self.m
        self.v += DT*a
        self.x += DT*self.v
        return self.x

class Plant01( Component ):
    def __init__( self, ):
        self.A = np.array([[1.,2.]   
                 ,[3.,4.]])
        self.B = np.array([[1.],[0.]])                 
        self.C = np.array([1.,0.]) 
        self.x = np.array([[0.],[0.]])
        self.y = 0
    def work( self, u ):
        U = np.array([[u],[u]]).T
        self.x = self.A.dot(self.x) + self.B*u
        self.y += self.C.dot(self.x)        
        return self.y

# ============================================================
# Filters and Actuators

class Identity( Component ):
    def work( self, x ): return x

class Limiter( Component ):
    def __init__( self, lo, hi ):
        self.lo = lo
        self.hi = hi

    def work( self, x ):
        return max( self.lo, min( x, self.hi ) )

class Discretizer( Component ):
    def __init__( self, binwidth ):
        self.binwidth = binwidth
    
    def work( self, u ):
        return self.binwidth*int( u/self.binwidth )

class Hysteresis( Component ):
    def __init__( self, threshold ):
        self.threshold = threshold
        self.prev = 0

    def work( self, u ):
        y = self.prev
        
        if abs(u - self.prev) > self.threshold:
            y = u
            self.prev = u

        return y    

class Integrator( Component ):
    def __init__( self ):
        self.data = 0

    def work( self, u ):
        self.data += u
        return DT*self.data

class FixedFilter( Component ):
    def __init__( self, n ):
        self.n = n
        self.data = []

    def work( self, x ):
        self.data.append(x)

        if len(self.data) > self.n:
            self.data.pop(0)

        return float(sum(self.data))/len(self.data)

class RecursiveFilter( Component ):
    def __init__( self, alpha ):
        self.alpha = alpha
        self.y = 0

    def work( self, x ):
        self.y = self.alpha*x + (1-self.alpha)*self.y
        return self.y

# ============================================================
# Setpoints

def impulse( t, t0 ):
    if abs(t-t0) < DT: return 1 # Floating point or integer time?
    return 0

def step( t, t0 ):
    if t >= t0: return 1
    return 0

def double_step( t, t0, t1 ):
    if t>=t0 and t<t1: return 1
    return 0

# def pulses( t, t0, tp ):
#     if t >= t0 and (t-t0)%tp == 0: return 1
#     return 0

def harmonic( t, t0, tp ):
    if t>=t0: return math.sin(2*math.pi*(t-t0)/tp)
    return 0

def relay( t, t0, tp ):
    if t>=t0:
        if math.ceil(math.sin(2*math.pi*(t-t0)/tp)) > 0:
            return 1
        else:
            return 0
    return 0

# ============================================================
# Loop functions

# def setpoint( t ):
#   return step( t, 0 )

def static_test( plant_ctor, ctor_args, umax, steps, repeats, tmax ):
    # Complete test for static process characteristic
    # From u=0 to umax taking steps steps, each one repeated repeats
    
    for i in range( 0, steps ):
        u = float(i)*umax/float(steps)

        for r in range( repeats ):
            p = apply( plant_ctor, ctor_args ) # this is: p = Plant( a, b, c )

            for t in range( tmax ):
                y = p.work(u)

            print u, y
            
    quit()

def step_response( setpoint, plant, tm=5000 ):
    for t in range( tm ):
        r = setpoint(t)  # This is the plant input, not really the setpoint!
        u = r
        y = plant.work( u )

        print t, t*DT, r, 0, u, u, y, y, plant.monitoring()
        
    quit()

def open_loop( setpoint, controller, plant, tm=5000 ):
    for t in range( tm ):
        r = setpoint(t)  # This is the controller input, not really the setpt!
        u = controller.work( r )
        y = plant.work( u )

        print t, t*DT, r, 0, u, u, y, y, plant.monitoring()

    quit()

def closed_loop( setpoint, controller, plant, tm=50, inverted=False,
                 actuator=Identity(), returnfilter=Identity() ):
    z = 0
    T = []
    Y = []
    U = []
    for t in range( tm ):
        r = setpoint(t)
        e = r - z
        if inverted == True: e = -e
        u = controller.work(e)
        v = actuator.work(u)
        y = plant.work(v)
        z = returnfilter.work(y)
        T.append(t)
        Y.append(y)
        U.append(y)

        #print t, t*DT, r, e, u, v, y, z, plant.monitoring()
    plt.plot(T,Y)
    #plt.plot(T,U)
    quit()

def bode_plot():
            # パラメータ設定
    m = 1
    c = 1
    k = 400

    A = np.array([[0, 1], [-k/m, -c/m]])
    B = np.array([[0], [k/m]])
    C = np.array([1, 0])
    D = np.array([0])

    s1 = signal.lti(A, B,   C, D)
    w, mag, phase = signal.bode(s1, np.arange(1, 500, 1)) 

    # プロット
    plt.figure(1)
    plt.subplot(211)
    plt.loglog(w, 10**(mag/20))
    plt.ylabel("Amplitude")
    plt.axis("tight")
    plt.subplot(212)
    plt.semilogx(w, phase)
    plt.xlabel("Frequency[Hz]")
    plt.ylabel("Phase[deg]")
    plt.axis("tight")
    plt.ylim(-180, 180)
    plt.savefig('../files/150613ABCD01.svg')
    # plt.show()    
# ============================================================

if __name__ == '__main__':

    def setpoint( t ):
        return 10*double_step( t, 100, 600 )

    p = Plant01()
    c = PidController( 0.45, 0.01 )

    closed_loop( setpoint, c, p, 15 )
#    (k, wn, z) = (1,1,0.1)          # 2次遅れ要素のパラメータ
#    G = tf([k*wn],[1,z*wn,wn**2])   # 伝達関数表現
#    nyquist(G)                      # ナイキスト線図の計算とプロット
#    plt.legend(["k=1"],1)           # 凡例
#    plt.grid()                      # グリッド
#    plt.show()                      # 結果表示
    #bode_plot()
