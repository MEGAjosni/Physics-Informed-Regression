# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:46:18 2021

@author: alboa
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
from tempfile import TemporaryFile
import deepxde as dde
from deepxde.backend import tf

import numpy as np
from scipy.integrate import odeint
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel2, LeastSquareModel, NoneNegativeLeastSquares

# SIR model parameters
simdays = 150   # compute ground truth
beta=0.5 
gamma=1/3

days_for_est = 4
t_true= np.arange(0, simdays)
varying_params_est = [] 

# parameters to be identified. There are as many betas as days


#beta function definition (arbitrary option. I chose sinwave here)
def mpfun(t,beta,gamma,simdays):
    if type(t) is int:
        return [0.05*np.sin(2*np.pi/simdays*t)+beta,gamma]
    else:
        return np.vstack([0.05*np.sin(2*np.pi/simdays*t)+beta,gamma*np.ones(len(t))]).T


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


mp = mpfun(t_true,beta,gamma,simdays)
C1true = mp[:,0]
C2true = gamma


# Generate true solution
x0_train = [9.99999833e-01, 1.66666667e-07, 0.00000000e+00] # Initial condition
x_true = SimulateModel2(t_true, x0_train, mp, model=SIR, realtime=True)

# Generate measurement data
dt = 0.1
days_for_est = 3
for day in range(days_for_est,simdays): 
    print(day)
    C1 = tf.Variable(0.5)
    C2 = tf.Variable(0.5)
    # trains a model for each day, taking the previous {days_for_est} number of days as training data
    # after each day save gamma and beta 

    
    t_test = np.arange(0, days_for_est,dt) # to be used for prediction
    t_test = t_test.reshape(len(t_test),-1)

    t_train = np.arange(0, days_for_est, dt) 
    x_train = SimulateModel2(t_train, x_true[day-days_for_est,:], mpfun(t_train+day,beta,gamma,simdays), model=SIR, realtime=True)
    
    t_train = t_train.reshape((len(t_train),1)) # reshape array to fit with DeepXDE
    
    
    # define time domain
    geom = dde.geometry.TimeDomain(0, days_for_est)
    
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
    
    
    # define FNN architecture and compile
    net = dde.maps.FNN([1] + [20] * 3 + [3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    #model.save(getcwd())
    
    # callbacks for storing results, and outputting the values of C1, C2 subject to parameter estimation (model calibration)
    fnamevar = "variables_SIR.dat"
    variable = dde.callbacks.VariableValue(
        [C1,C2], 
        period=1,
        filename=fnamevar
    )
    
    losshistory, train_state = model.train(epochs=40000, callbacks=[variable])
    
    # reopen saved data using callbacks in fnamevar 
    lines = open(fnamevar, "r").readlines()
    
    # read output data in fnamevar (this line is a long story...)
    Chat = np.array([np.fromstring(min(re.findall(re.escape('[')+"(.*?)"+re.escape(']'),line), key=len), sep=',') for line in lines])
    varying_params_est.append([Chat[-1,0],Chat[-1,1]]) # Assume best is found at end of training 
  
np.array([varying_params_est])
np.savetxt('pinn_params_SIR.out',varying_params_est,delimiter = ',')
