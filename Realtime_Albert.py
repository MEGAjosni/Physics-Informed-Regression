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
simdays = 70  # compute ground truth
beta=0.4 
gamma=1/3
dt = 0.1
t_true= np.arange(0, simdays)
varying_params_est = [] 
avg__true_params = []

# parameters to be identified. There are as many betas as days


#beta function definition (arbitrary option. I chose sinwave here)
def mpfun(t,beta,gamma,simdays):
    if type(t) is int:
        return [0.05*np.sin(2*np.pi/simdays*t)+beta,gamma]
        #return[beta,gamma]
    else:
        return np.vstack([0.05*np.sin(2*np.pi/simdays*t)+beta,gamma*np.ones(len(t))]).T
        #return np.vstack([beta*np.ones(len(t)),gamma*np.ones(len(t))]).T

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


# generate mp (with high resolution of dt, over entire domain)
t_mp= np.arange(0, simdays,dt)
mp = mpfun(t_mp,beta,gamma,simdays/2)


# Generate true solution
x0_train = [1-0.005, 0.005, 0.00000000e+00] # Initial condition
x_true = SimulateModel2(t_true, x0_train, mp, model=SIR, realtime=True)

# Generate measurement data

for day in range(1,simdays): 
    
    days_for_est = 4
    if days_for_est > day:
        days_for_est = day
        
    # start of training for current day's estimate    
    t_0 = day-days_for_est
    
    print(day)
    C1 = tf.Variable(0.5)
    C2 = tf.Variable(0.5)
    # trains a model for each day, taking the previous {days_for_est} number of days as training data
    

    # define train data for relevant days used as training for this particular day
    t_test = np.arange(t_0,day,dt) 
    t_test = t_test.reshape(len(t_test),-1)
    t_train = np.arange(t_0,day, dt)
    # take slice of mp matrix as current mp values 
    mp_day = mp[int(t_0/dt):int(day/dt)]
    
    x_train = SimulateModel2(t_train, x_true[t_0,:], mp_day, model=SIR, realtime=True)
    t_train = t_train.reshape((len(t_train),1)) # reshape array to fit with DeepXDE
    
    # define time domain
    geom = dde.geometry.TimeDomain(t_0, day)
    
    # Initial conditions
    ic1 = dde.IC(geom, lambda X: x_true[t_0,0], boundary, component=0) # S
    ic2 = dde.IC(geom, lambda X: x_true[t_0,1], boundary, component=1) # I
    ic3 = dde.IC(geom, lambda X: x_true[t_0,2], boundary, component=2) # R
    
    
    # point boundary conditions
    observe_t = t_train
    observe_z = x_train
    observe_z0 = dde.PointSetBC(observe_t, observe_z[:, 0:1], component=0)
    observe_z1 = dde.PointSetBC(observe_t, observe_z[:, 1:2], component=1)
    observe_z2 = dde.PointSetBC(observe_t, observe_z[:, 2:], component=2)
    
    # define data object
    data = dde.data.PDE(
        geom,
        SIR_system,
        [ic1, ic2, ic3, observe_z0, observe_z1, observe_z2],
        num_domain = int(days_for_est/dt),
        num_boundary=2,
        
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
    
    # train model
    losshistory, train_state = model.train(epochs=50000, callbacks=[variable])
    
    # reopen saved data using callbacks in fnamevar 
    lines = open(fnamevar, "r").readlines()
    
    # read output data in fnamevar (this line is a long story...)
    Chat = np.array([np.fromstring(min(re.findall(re.escape('[')+"(.*?)"+re.escape(']'),line), key=len), sep=',') for line in lines])
    
    # log parameters from best epoch, and output during training
    varying_params_est.append(Chat[train_state.best_step])
    avg_beta = sum(mp_day[:,0])/(len(mp_day[:,0]))
    avg_gamma = sum(mp_day[:,1])/len(mp_day[:,1])
    avg__true_params.append([avg_beta,avg_gamma])
    print("----   Estimated -- Actual avg")
    print("beta  ",Chat[train_state.best_step][0],"     ",avg_beta)
    print("gamma ",Chat[train_state.best_step][1],"     ",avg_gamma)
    #print(mp_day)
    
# save estimated parameters    
np.savetxt('pinn_params_SIR3.out',varying_params_est,delimiter = ',')


# plot against actual parameters 
beta_est = [0]*len(varying_params_est)
gamma_est = [0]*len(varying_params_est)

for i in range(len(varying_params_est)):
    beta_est[i] = (varying_params_est[i])[0]
    gamma_est[i] = (varying_params_est[i])[1]

t_params = np.arange(1,simdays)
plt.plot(t_mp,mp)
plt.scatter(t_params,beta_est)
plt.scatter(t_params,gamma_est)
plt.legend(['beta','gamma','beta est', 'gamma est'],loc='upper right')
plt.title('estimating varying parameters, 4 day training per day')
plt.xlabel('day')
plt.ylim((0,1))
plt.show()

# plot prediction vs true data (true being data created with target parameters)
x_pred = SimulateModel2(t_params, x0_train, varying_params_est, model = SIR, realtime=(True))
plt.scatter(t_params,x_pred[:,0])
plt.scatter(t_params,x_pred[:,1])
plt.scatter(t_params,x_pred[:,2])
plt.plot(t_true,x_true)

