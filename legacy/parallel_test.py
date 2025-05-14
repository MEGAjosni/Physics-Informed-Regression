# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 20:11:22 2021

@author: alboa
"""


import io
import re
from os import getcwd
import matplotlib.pyplot as plt
import numpy as np
import requests
from tempfile import TemporaryFile


import multiprocessing as mpp

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

'''
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
'''
def parallel_fun(day):
    arr[day] = day
    ## code ##
    return 1


# Generate measurement data
dt = 0.1
days_for_est = 3

arr = [0]*simdays
# parallel loop:
pool = mpp.Pool(mpp.cpu_count())
results_objects = [pool.apply_async(func=(parallel_fun), args=(day))for day in range(1,simdays)]


