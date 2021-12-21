# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 12:49:17 2021

@author: alboa
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import re
import tikzplotlib
from os import getcwd
import matplotlib.pyplot as plt
import numpy as np
import requests
import deepxde as dde
from deepxde.backend import tf
import numpy as np

# user made imports
import models as models
import sim_functions as sf

### Data ###

file_string = "S3I3R_"

# S3I3R model parameters
tfinal =  7*6  # total days
t0predict = 7*4# timie limit for traininig data, extrapolate from this time
beta = 0.5
gamma1 = 1/3 
gamma2 = 1/20
gamma3 = 1/20
phi1 = 1/20
phi2 = 1/20
theta = 1/10
tau = 0
mp = [beta, gamma1, gamma2, gamma3, phi1, phi2, theta, tau]

# parameters to be estimated (only 4 of 8)
beta_var = tf.Variable(1.0)
phi1_var = tf.Variable(1.0)
phi2_var = tf.Variable(1.0)
theta_var = tf.Variable(1.0)

# Generate true solution
dt = .25 # time step
t_true = np.arange(0, tfinal, dt)
x0_train = [0.999,0.001,0.0,0.0,0.0,0.0,0.0] # Initial condition
x_true = sf.SimulateModel(t_true, x0_train, mp,model=models.S3I3R) # why compute here, when overriding below?

# Generate data 
dt = .1 # time step
t_test = np.arange(dt/2.0, tfinal, dt) # to be used for prediction
x_true = sf.SimulateModel(t_test, x0_train, mp,model=models.S3I3R)
t_test = t_test.reshape(len(t_test),-1)
t_train = np.arange(0, t0predict, dt)
x_train = sf.SimulateModel(t_train, x0_train, mp,model=models.S3I3R)
t_train = t_train.reshape((len(t_train),1)) # reshape array to fit with DeepXDE


### Defining boundary function and ode with deepxde framework ###


def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

# S3I3R model definition (using _var parameters defined above)
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



### Define geom and initial/boundary conditions ###


# define time domain
geom = dde.geometry.TimeDomain(0, tfinal)

# Initial conditions
ic1 = dde.IC(geom, lambda X: x0_train[0], boundary, component=0) # S
ic2 = dde.IC(geom, lambda X: x0_train[1], boundary, component=1) # I1
ic3 = dde.IC(geom, lambda X: x0_train[2], boundary, component=2) # I2
ic4 = dde.IC(geom, lambda X: x0_train[3], boundary, component=3) # I3
ic5 = dde.IC(geom, lambda X: x0_train[4], boundary, component=4) # R1
ic6 = dde.IC(geom, lambda X: x0_train[5], boundary, component=5) # R2
ic7 = dde.IC(geom, lambda X: x0_train[6], boundary, component=6) # R3

# Get the training data (synthetic data, but could be any data)
observe_t = t_train
observe_X = x_train

# define boundary conditions for each compartment.
observe_S = dde.PointSetBC(observe_t, observe_X[:, 0:1], component=0)
observe_I1 = dde.PointSetBC(observe_t, observe_X[:, 1:2], component=1)
observe_I2 = dde.PointSetBC(observe_t, observe_X[:, 2:3], component=2)
observe_I3 = dde.PointSetBC(observe_t, observe_X[:, 3:4], component=3)
observe_R1 = dde.PointSetBC(observe_t, observe_X[:, 4:5], component=4)
observe_R2 = dde.PointSetBC(observe_t, observe_X[:, 5:6], component=5)
observe_R3 = dde.PointSetBC(observe_t, observe_X[:, 6:], component=6)

# define data object
data = dde.data.PDE(
    geom,
    S3I3R_system,
    [ic1, ic2, ic3, ic4, ic5, ic6, ic7,observe_S, observe_I1, observe_I2, observe_I3, observe_R1, observe_R2, observe_R3],
    num_domain=int(t0predict/dt), 
    num_boundary=2,
    anchors=t_test
)
### initialize model, and train ###


# define FNN architecture and compile
net = dde.maps.FNN([1] + [40] * 8 + [7], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

# callbacks for storing results, and outputting the values of C1, C2 subject to parameter estimation (model calibration)
fnamevar = "variables_expanded_test.dat"
variable = dde.callbacks.VariableValue(
    [beta_var,phi1_var,phi2_var,theta_var], 
    period=1,
    filename=fnamevar
)

losshistory, train_state = model.train(epochs=70000, callbacks=[variable])


### Post training processing ###


# reopen saved data using callbacks in fnamevar 
lines = open(fnamevar, "r").readlines()

# read output data in fnamevar (this line is a long story...)
Chat = np.array([np.fromstring(min(re.findall(re.escape('[')+"(.*?)"+re.escape(']'),line), key=len), sep=',') for line in lines])
l,c = Chat.shape


# plot parameter convergance by epoch
plt.plot(range(l),Chat[:,0],'r-')
plt.plot(range(l),Chat[:,1],'k-')
plt.plot(range(l),Chat[:,2],'b-')
plt.plot(range(l),Chat[:,3],'y-')
plt.plot(range(l),np.ones(Chat[:,0].shape)*beta,'r--')
plt.plot(range(l),np.ones(Chat[:,1].shape)*phi1,'k--')
plt.plot(range(l),np.ones(Chat[:,2].shape)*phi2,'g--')
plt.plot(range(l),np.ones(Chat[:,3].shape)*theta,'y--')
plt.legend(['beta est','phi1 est','phi2 est', 'theta est','beta','phi1','phi2','theta'])
plt.xlabel('Epoch')
tikzplotlib.save("S3I3R_const_v1_params.tex")
plt.show()



# test parameters with prediction
yhat = model.predict(t_test)
plt.plot(t_test,x_true[:,1],'k',observe_t, observe_X[:,1],'-',t_test, yhat[:,1],'--')
plt.plot(t_test,x_true[:,2],'k',observe_t, observe_X[:,2],'-',t_test, yhat[:,2],'--')
plt.plot(t_test,x_true[:,3],'k',observe_t, observe_X[:,3],'-',t_test, yhat[:,3],'--')
plt.legend(['I1 train','I1 true','I1 pred','I2 train','I2 true','I2 pred','I3 train','I3 true','I3 pred'])
plt.ylabel('Population fraction')
plt.xlabel('Days')
plt.title('Training data vs. PINN solution')
plt.xlim(0,tfinal)
tikzplotlib.save("S3I3R_const_v1_I123.tex")
plt.show()

