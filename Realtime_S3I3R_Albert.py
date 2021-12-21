# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 11:36:14 2021

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

import tikzplotlib

import numpy as np
from scipy.integrate import odeint
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel2, LeastSquareModel, NoneNegativeLeastSquares

# SIR model parameters
simdays = 28*2  # compute ground truth
beta = 0.5
gamma1 = 1/3 
gamma2 = 1/20
gamma3 = 1/20
phi1 = 1/20
phi2 = 1/20
theta = 1/10
tau = 0.001
dt = 0.1
t_true= np.arange(0, simdays)
varying_params_est = [] 
avg__true_params = []


#beta function definition (arbitrary option. I chose sinwave here)
def mpfun(t,beta,gamma1,gamma2,gamma3,phi1,phi2,theta,tau,simdays):
    if type(t) is int:
        return [0.05*np.sin(2*np.pi/simdays*t)+beta,gamma1,gamma2,gamma3,phi1,phi2,theta,tau]
        #return[beta,gamma]
    else:
        return np.vstack([0.05*np.sin(2*np.pi/simdays*t)+beta,
                          gamma1*np.ones(len(t)),
                          gamma2*np.ones(len(t)),
                          gamma3*np.ones(len(t)),
                          phi1*np.ones(len(t)),
                          phi2*np.ones(len(t)),
                          theta*np.ones(len(t)),
                          tau*np.ones(len(t))]).T
        #return np.vstack([beta*np.ones(len(t)),gamma*np.ones(len(t))]).T

def boundary(_, on_initial):
    return on_initial


# S3I3R model definition
def S3I3R_system(t,X):
    # note, vacination not implemented due to nan errors corrupting training. 
    
    S,I1,I2,I3,R1,R2,R3 = X[:,0:1], X[:,1:2], X[:,2:3],X[:,3:4],X[:,4:5],X[:,5:6],X[:,6:]
    dS_t = dde.grad.jacobian(X, t, i=0)
    dI1_t = dde.grad.jacobian(X, t, i=1)
    dI2_t = dde.grad.jacobian(X, t, i=2)
    dI3_t = dde.grad.jacobian(X, t, i=3)
    dR1_t = dde.grad.jacobian(X, t, i=4)
    dR2_t = dde.grad.jacobian(X, t, i=5)
    dR3_t = dde.grad.jacobian(X, t, i=6)
    
    # return rhs-lhs in ode_system
    return [
        dS_t - (-(beta_var*I1)*S),
        dI1_t - (beta_var*I1*S - (gamma1 + phi1_var)*I1),
        dI2_t - (phi1_var * I1 -(gamma2+ phi2_var) * I2),
        dI3_t - (phi2_var * I2 -(gamma3 + theta_var) * I3),
        dR1_t - (gamma1 * I1 + gamma2 * I2 + gamma3 * I3),
        dR2_t - 0,
        dR3_t - theta_var * I3]


# generate mp (with high resolution of dt, over entire domain)
t_mp= np.arange(0, simdays,dt)
mp = mpfun(t_mp,beta,gamma1,gamma2,gamma3,phi1,phi2,theta,tau,simdays)


# Generate true solution
x0_train = [0.99, 0.001, 0,0,0,0,0] # Initial condition
x_true = SimulateModel2(t_true, x0_train, mp, model=S3I3R, realtime=True)

# Generate measurement data

# plot prediction vs true data (true being data created with target parameters)
t_params = np.arange(1,simdays)


for day in range(1,simdays): 
    
    days_for_est = 7
    if days_for_est > day:
        days_for_est = day
        
    # start of training for current day's estimate    
    t_0 = day-days_for_est
    
    print(day)
    
    # trains a model for each day, taking the previous {days_for_est} number of days as training data
    beta_var = tf.Variable(0.5)
    phi1_var = tf.Variable(0.5)
    phi2_var = tf.Variable(0.5)
    theta_var = tf.Variable(0.5)


    # define train data for relevant days used as training for this particular day
    t_test = np.arange(t_0,day,dt) 
    t_test = t_test.reshape(len(t_test),-1)
    t_train = np.arange(t_0,day, dt)
    # take slice of mp matrix as current mp values 
    mp_day = mp[int(t_0/dt):int(day/dt)]
    
    x_train = SimulateModel2(t_train, x_true[t_0,:], mp_day, model=S3I3R, realtime=True)
    t_train = t_train.reshape((len(t_train),1)) # reshape array to fit with DeepXDE
    
    # define time domain
    geom = dde.geometry.TimeDomain(t_0, day)
    
    # Initial conditions
    ic1 = dde.IC(geom, lambda X: x_true[t_0,0], boundary, component=0) # S
    ic2 = dde.IC(geom, lambda X: x_true[t_0,1], boundary, component=1) # I1
    ic3 = dde.IC(geom, lambda X: x_true[t_0,2], boundary, component=2) # I2
    ic4 = dde.IC(geom, lambda X: x_true[t_0,3], boundary, component=3) # I3
    ic5 = dde.IC(geom, lambda X: x_true[t_0,4], boundary, component=4) # R1
    ic6 = dde.IC(geom, lambda X: x_true[t_0,5], boundary, component=5) # R2
    ic7 = dde.IC(geom, lambda X: x_true[t_0,6], boundary, component=6) # R3
    
    
    # point boundary conditions
    observe_t = t_train
    observe_X = x_train
    observe_S = dde.PointSetBC(observe_t, observe_X[:, 0:1], component=0)  # S
    observe_I1 = dde.PointSetBC(observe_t, observe_X[:, 1:2], component=1) # I1
    observe_I2 = dde.PointSetBC(observe_t, observe_X[:, 2:3], component=2) # I2
    observe_I3 = dde.PointSetBC(observe_t, observe_X[:, 3:4], component=3) # I3
    observe_R1 = dde.PointSetBC(observe_t, observe_X[:, 4:5], component=4) # R1
    observe_R2 = dde.PointSetBC(observe_t, observe_X[:, 5:6], component=5) # R2 
    observe_R3 = dde.PointSetBC(observe_t, observe_X[:, 6:], component=6)  # R3
    
    # define data object
    data = dde.data.PDE(
        geom,
        S3I3R_system,
        [ic1, ic2, ic3, ic4, ic5, ic6, ic7,observe_S, observe_I1, observe_I2, observe_I3, observe_R1, observe_R2, observe_R3],
        num_domain = int(days_for_est/dt),
        num_boundary=2,
        
    )
    
    
    # define FNN architecture and compile
    net = dde.maps.FNN([1] + [40] * 8 + [7], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001)
    #model.save(getcwd())
    
    # callbacks for storing results, and outputting the values of C1, C2 subject to parameter estimation (model calibration)
    fnamevar = "variables_S3I3R.dat"
    variable = dde.callbacks.VariableValue(
        [beta_var,phi1_var,phi2_var,theta_var], 
        period=1,
        filename=fnamevar
    )
    
    # train model
    losshistory, train_state = model.train(epochs=60000, callbacks=[variable])
    
    # reopen saved data using callbacks in fnamevar 
    lines = open(fnamevar, "r").readlines()
    
    # read output data in fnamevar (this line is a long story...)
    Chat = np.array([np.fromstring(min(re.findall(re.escape('[')+"(.*?)"+re.escape(']'),line), key=len), sep=',') for line in lines])
    
    # log parameters from best epoch, and output during training
    varying_params_est.append(Chat[train_state.best_step])
    avg_beta = sum(mp_day[:,0])/(len(mp_day[:,0]))
    avg_phi1 = sum(mp_day[:,4])/len(mp_day[:,4])
    avg_phi2 = sum(mp_day[:,5])/len(mp_day[:,5])
    avg_theta = sum(mp_day[:,6])/len(mp_day[:,6])
    print("----   Estimated -- Actual avg")
    print("Beta ",Chat[train_state.best_step][0],"     ",avg_beta)
    print("Phi1 ",Chat[train_state.best_step][1],"     ",avg_phi1)
    print("Phi2 ",Chat[train_state.best_step][2],"     ",avg_phi2)
    print("theta",Chat[train_state.best_step][3],"     ",avg_theta)
    print(mp_day)
    
# save estimated parameters    
np.savetxt('varying_parameters_S3I3R_v5.out',varying_params_est,delimiter = ',')


# plot against actual parameters 
beta_est = [0]*len(varying_params_est)
phi1_est = [0]*len(varying_params_est)
phi2_est = [0]*len(varying_params_est)
theta_est = [0]*len(varying_params_est)

for i in range(len(varying_params_est)):
    beta_est[i] = (varying_params_est[i])[0]
    phi1_est[i] = (varying_params_est[i])[1]
    phi2_est[i] = (varying_params_est[i])[2]
    theta_est[i] = (varying_params_est[i])[3]


fig = plt.figure()
ax = plt.subplot(111)
plt.plot(t_mp,mp[:,0])
plt.plot(t_mp,mp[:,4])
plt.plot(t_mp,mp[:,5])
plt.plot(t_mp,mp[:,6])
plt.scatter(t_params,beta_est)
plt.scatter(t_params,phi1_est)
plt.scatter(t_params,phi2_est)
plt.scatter(t_params,theta_est)
plt.legend(['beta','phi1','phi2', 'theta','beta est','phi1 est','phi2 est', 'theta est'],loc='upper left',bbox_to_anchor=(0.5,1.05),ncol=2)
plt.xlabel('day')
plt.ylim((-0.2,1))
tikzplotlib.save("S3I3R_v5_5_days_4_est_larger.tex",)
plt.show()

# plot prediction vs true data (true being data created with target parameters)
#x_pred = SimulateModel2(t_params, x0_train, varying_params_est, model = S3I3R, realtime=(True))
plt.scatter(t_true,x_true [:,0])
plt.scatter(t_true,x_true [:,1])
plt.scatter(t_true,x_true [:,2])
plt.scatter(t_true,x_true [:,3])
plt.scatter(t_true,x_true [:,4])
plt.scatter(t_true,x_true [:,5])
plt.scatter(t_true,x_true [:,6])
plt.plot(t_true,x_true)
