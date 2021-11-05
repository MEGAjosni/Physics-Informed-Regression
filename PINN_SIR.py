# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 09:35:04 2021

@author: Marcu
"""

"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

import deepxde as dde
import numpy as np
from scipy.integrate import solve_ivp
import basic_ivp_funcs as b_ivp
import matplotlib.pyplot as plt

#Seed and precision
np.random.seed(1234)
dde.config.real.set_float32()




def derivative(
        t,
        x: list,  # Vector to compute derivative of
):
    S, I, R, beta, gamma = x.transpose()
    N = S[0]+I[0]+R[0]

    dX = [
        - beta * S * I / N,
        I * (beta * S / N - gamma),
        gamma * I
    ]

    return np.array(dX)

def ode_system(t, x):
    ### ode system #####
    # calculates the residuals corresponding to for 
    # each ode in the system
    
    
    # x = [S,I,R, beta, gamma]
    S, I, R, beta, gamma = x.transpose()
    N = S[0]+I[0]+R[0]
    
    S, I, R = x[:, 0], x[:, 1], x[:, 2]
    dS = dde.grad.jacobian(x, t, i=0)
    dI = dde.grad.jacobian(x, t, i=1)
    dR = dde.grad.jacobian(x, t, i=2)
    
    
    return [dS+beta*I*S/N, dI - beta*I*S/N + gamma * I, -gamma*I]


def boundary(_, on_initial):
    return on_initial


def func(t):
    sol = solve_ivp(derivative,t_span = [0,t[-1]], y0 = [10000, 30, 0],t_eval=t)
    
    return np.reshape(sol.y,(len(t)*3,1))

"""
def ic_1():
    return np.array([10000])
def ic_2():
    return np.array([30])
def ic_3():
    return np.array([0])

geom = dde.geometry.TimeDomain(0, 100)
ic1 = dde.IC(geom, ic_1, boundary, component=0)
ic2 = dde.IC(geom, ic_2, boundary, component=1)
ic3  =dde.IC(geom, ic_3, boundary, component=2)
data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3], 35, 2, solution=func, num_test=100)

layer_size = [1] + [50] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
losshistory, train_state = model.train(epochs=20000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)
"""