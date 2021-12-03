# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 00:17:49 2021

@author: Marcus
"""

import matplotlib.pyplot as plt
import numpy as np
import get_synth_data as gsd
import paramest_functions_OLS as pest_OLS
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel2, LeastSquareModel, NoneNegativeLeastSquares

#initialise duration of simulation
simdays = 150
t1 = 0
t2 = t1+simdays


############ Real time estimations ################################
over_time = True #set "True" if estimations are needed in real time
overshoot = 4 #the amount of previous days included for parameter estimations in real time


#initialize IC and parameters for the syntetic data
X0 = [9.99999833e-01, 1.66666667e-07, 0.00000000e+00] #initial conditions
beta= 0.5  #rate of transmission
gamma = 1/3     #rate of recovery
noise_var = 0 #variance of added noise

t = np.arange(simdays)

############ Which parameters to find ####################################
pass_beta = None #"None" if you need a beta estimation, otherwise put a "beta" value
pass_gamma = None #"None" if you need a gamma estimation, otherwise put a "gamma" value


#### plots ##############
include_params = True #include parameters in plot



#Generate synthetic data or use real data

mp = np.array([[0.05*np.sin(2*np.pi/simdays*i)+beta,gamma] for i in range(simdays)])

X_syn = SimulateModel2(t, X0, mp, model=SIR, realtime=over_time)


    #real time parameter estimation from data
mp_est = pest_OLS.SIR_params_over_time_OLS(
        t1 = t1,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn,
        beta = pass_beta,
        gamma = pass_gamma)


#simulation using retrieved parameters
X_est = SimulateModel2(t[overshoot:], X_syn[overshoot, :], mp_est, model=SIR, realtime=over_time)


#plots
n = 5
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.scatter(t[::n],X_syn[::n,1])
ax.plot(t[overshoot:],X_est[:,1])
ax.legend("S",loc="upper left")

if include_params:
    ax2.plot(mp_est[:,0], c = "r")
    ax2.plot(mp_est[:,1], c = "tab:purple")
    if over_time:
        ax2.plot(t,mp[:,0],"--",c = "r")
        ax2.plot(t,mp[:,1],"--",c = "tab:purple")
    else:
       ax2.plot(t,beta*np.ones(len(t)),"--",c = "r")
       ax2.plot(t,gamma*np.ones(len(t)),"--",c = "tab:purple")
    ax2.set_ylim(0,0.30)
    ax2.legend(["b est","g est","b true", "g true"],loc = "upper right")


