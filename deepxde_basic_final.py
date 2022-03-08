# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 00:23:04 2021

@author: Marcu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import re
from os import getcwd

import matplotlib.pyplot as plt
import numpy as np
import requests

import deepxde as dde
from deepxde.backend import tf

import numpy as np
from scipy.integrate import odeint
import tikzplotlib

# SIR model parameters
tfinal = 80   # compute ground truth
t0predict = 50 # timie limit for traininig data, extrapolate from this time
b=0.5 
k=1/3

# parameters to be identified
C1 = tf.Variable(1.0)
C2 = tf.Variable(1.0)

C1true = b
C2true = k

# SIR model definition
def SIR(z, t):
    return [
      -b*z[0]*z[1]  ,
       b*z[0]*z[1] - k*z[1] ,
       k*z[1] 
    ]

# Generate true solution
dt = .25 # time step
t_true = np.arange(0, tfinal, dt)
x0_train = [1-0.0001, 0.0001, 0.00000000e+00] # Initial condition

# Generate measurement data
dt = .1 # time step
t_test = np.arange(dt/2.0, tfinal, dt) # to be used for prediction
x_true = odeint(SIR, x0_train, t_test)
t_test = t_test.reshape(len(t_test),-1)
t_train = np.arange(0, t0predict, dt)
x_train = odeint(SIR, x0_train, t_train)
t_train = t_train.reshape((len(t_train),1)) # reshape array to fit with DeepXDE

def boundary(_, on_initial):
    return on_initial


# SIR model definition
def SIR_system(t,z):
    z0, z1, z2 = z[:,0:1], z[:,1:2], z[:,2:]
    dz0_t = dde.grad.jacobian(z, t, i=0)
    dz1_t = dde.grad.jacobian(z, t, i=1)
    dz2_t = dde.grad.jacobian(z, t, i=2)
    return [
      dz0_t - ( -C1*z0*z1 )  ,
      dz1_t - ( C1*z0*z1 - C2*z1 ),
      dz2_t - ( C2*z1 )
    ]

# define time domain
geom = dde.geometry.TimeDomain(0, tfinal)

# Initial conditions
ic1 = dde.IC(geom, lambda X: x0_train[0], boundary, component=0) # S
ic2 = dde.IC(geom, lambda X: x0_train[1], boundary, component=1) # I
ic3 = dde.IC(geom, lambda X: x0_train[2], boundary, component=2) # R

# Get the training data (synthetic data, but could be any data)
observe_t = t_train
observe_z = x_train
#observe_t, ob_y = gen_traindata()
observe_z0 = dde.PointSetBC(observe_t, observe_z[:, 0:1], component=0)
observe_z1 = dde.PointSetBC(observe_t, observe_z[:, 1:2], component=1)
observe_z2 = dde.PointSetBC(observe_t, observe_z[:, 2:], component=2)

# define data object
data = dde.data.PDE(
    geom,
    SIR_system,
    [ic1, ic2, ic3, observe_z0, observe_z1, observe_z2],
    num_domain=800,
    num_boundary=2,
    anchors=t_test
)

plt.plot(observe_t, observe_z)
plt.xlabel('Time')
plt.legend(['S','I','R'])
plt.title('Training data')
plt.show()

# define FNN architecture and compile
net = dde.maps.FNN([1] + [20] * 3 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
#model.save(getcwd())

# callbacks for storing results, and outputting the values of C1, C2 subject to parameter estimation (model calibration)
fnamevar = "variables_SIR_temp.dat"
variable = dde.callbacks.VariableValue(
    [C1,C2], 
    period=1,
    filename=fnamevar
)

losshistory, train_state = model.train(epochs=70000, callbacks=[variable])

# reopen saved data using callbacks in fnamevar 
lines = open(fnamevar, "r").readlines()

# read output data in fnamevar (this line is a long story...)
Chat = np.array([np.fromstring(min(re.findall(re.escape('[')+"(.*?)"+re.escape(']'),line), key=len), sep=',') for line in lines])

l,c = Chat.shape

n = 500
epoch = np.arange(0,l)
plt.plot(epoch[::n],Chat[::n,0],'r-')
plt.plot(epoch[::n],Chat[::n,1],'k-')
plt.plot(epoch[::n],np.ones(Chat[::n,0].shape)*C1true,'r--')
plt.plot(epoch[::n],np.ones(Chat[::n,1].shape)*C2true,'k--')
plt.legend(['C1hat','C2hat','True C1','True C2'],loc = "right")
plt.xlabel('Epoch')
tikzplotlib.save("SIR_const_v1_params.tex")
plt.show()


yhat = model.predict(t_test)

plt.plot(t_test,x_true,'k',observe_t, observe_z,'-',t_test, yhat,'--')
plt.ylabel('Persons/Population')
plt.xlabel('Time [days]')
plt.legend(['$S_{true}$','$I_{true}$','$R_{true}$','$S_{data}$','$I_{data}$','$R_{data}$','$S_h$','$I_h$','$R_h$'])
plt.title('Training data vs. PINN solution')
tikzplotlib.save("SIR_const_v1.tex")
plt.show()
