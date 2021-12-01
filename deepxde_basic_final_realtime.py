# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 00:57:51 2021

@author: Marcu
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import io
import re
from os import getcwd
import matplotlib.pyplot as plt
import numpy as np
import requests

import deepxde as dde
#from deepxde.backend import tf

import numpy as np
from scipy.integrate import odeint
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel, LeastSquareModel, NoneNegativeLeastSquares

# SIR model parameters
simdays = 200   # compute ground truth
beta=0.5 
gamma=1/3
t0predict = 200 # timie limit for traininig data, extrapolate from this time
t_true= np.arange(0, simdays)


# parameters to be identified. There are as many betas as days
C1 = tf.Variable(np.ones(simdays))
C2 = tf.Variable(1)

#beta function definition (arbitrary option. I chose sinwave here)
def mpfun(t,beta,gamma,simdays):
    return np.vstack([0.05*np.sin(2*np.pi/simdays*t)+beta,gamma*np.ones(len(t))]).T

mp = mpfun(t_true,beta,gamma,simdays)
C1true = mp[:,0]
C2true = gamma


# Generate true solution
x0_train = [9.99999833e-01, 1.66666667e-07, 0.00000000e+00] # Initial condition
x_true = SimulateModel(t_true, x0_train, mp, model=SIR, realtime=True)

# Generate measurement data
dt = 0.1
t_test = np.arange(0, simdays,dt) # to be used for prediction
x_true = SimulateModel(t_test, x0_train, mpfun(t_test,beta,gamma,simdays), model=SIR, realtime=True)
t_test = t_test.reshape(len(t_test),-1)
t_train = np.arange(0, t0predict, dt)
x_train = SimulateModel(t_train, x0_train, mpfun(t_train,beta,gamma,simdays), model=SIR, realtime=True)
t_train = t_train.reshape((len(t_train),1)) # reshape array to fit with DeepXDE

def boundary(_, on_initial):
    return on_initial


# SIR model definition
def SIR_system(t,z):
    day = tf.cast(tf.round(t),tf.int64).numpy()
    z0, z1, z2 = z[:,0:1], z[:,1:2], z[:,2:]
    dz0_t = dde.grad.jacobian(z, t, i=0)
    dz1_t = dde.grad.jacobian(z, t, i=1)
    dz2_t = dde.grad.jacobian(z, t, i=2)
    return [
      dz0_t - ( -tf.gather(C1,day)*z0*z1 )  ,
      dz1_t - ( tf.gather(C1,day)*z0*z1 - tf.gather(C2,day)*z1 ),
      dz2_t - ( tf.gather(C2,day)*z1 )
    ]   

# define time domain
geom = dde.geometry.TimeDomain(0, simdays)

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
model.save(getcwd())


# callbacks for storing results, and outputting the values of C1, C2 subject to parameter estimation (model calibration)
fnamevar = "variables_SIR_realtime.dat"
variable = dde.callbacks.VariableValue(
    [C1,C2], 
    period=1,
    filename=fnamevar
)

losshistory, train_state = model.train(epochs=1000, callbacks=[variable])

# reopen saved data using callbacks in fnamevar 
lines = open(fnamevar, "r").readlines()

# read output data in fnamevar (this line is a long story...)
Chat = np.array([np.fromstring(min(re.findall(re.escape('[')+"(.*?)"+re.escape(']'),line), key=len), sep=',') for line in lines])

l,c = Chat.shape

plt.plot(range(l),Chat[:,0],'r-')
plt.plot(range(l),Chat[:,1],'k-')
plt.plot(range(l),np.ones(Chat[:,0].shape)*C1true,'r--')
plt.plot(range(l),np.ones(Chat[:,1].shape)*C2true,'k--')
plt.legend(['C1hat','C2hat','True C1','True C2'],loc = "right")
plt.xlabel('Epoch')
plt.show()

yhat = model.predict(t_test)

plt.plot(t_test,x_true,'k',observe_t, observe_z,'-',t_test, yhat,'--')
plt.ylabel('Persons/Population')
plt.xlabel('Time [days]')
plt.legend(['$S_{true}$','$I_{true}$','$R_{true}$','$S_{data}$','$I_{data}$','$R_{data}$','$S_h$','$I_h$','$R_h$'])
plt.title('Training data vs. PINN solution')
plt.show()