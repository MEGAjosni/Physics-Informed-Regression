# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:26:34 2022

Synthetic data

@author: Marcus
"""

import matplotlib.pyplot as plt
import numpy as np
import paramest_functions_OLS as pest_OLS
import scipy.integrate
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel2, LeastSquareModel, NoneNegativeLeastSquares

simdays = 100
t1 = 0
t2 = t1+simdays
overshoot = 1;

X0 = [1-0.005, 0.005, 0.0] #initial conditions
beta= 0.5  #rate of transmission
gamma = 1/3     #rate of recovery
noise_var = 0 #variance of added noise

t = np.arange(simdays)

def betafun1(day,simdays,beta):
    return 0.1*np.sin(4*np.pi/simdays*day)+beta
    
def betafun2(day, simdays, beta):
    if day < simdays/2:
        return beta/2
    else:
        return 3*beta/2

def betafun3(day, simdays, beta):
    if day < simdays/2:
        return 2*(beta-beta/2)/simdays*day+beta/2
    else:
        return beta            

#define model parameters
mp = np.array([[betafun2(i,simdays,beta),gamma] for i in range(simdays)])    
  
#compute  syntgetic data  
X_syn = SimulateModel2(t, X0, mp, model=SIR, realtime=True)

#estimate parameters
mp_est = pest_OLS.SIR_params_over_time_OLS(
        t1 = t1,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn,
        beta = None,
        gamma = gamma)

#recreate solution using estimated parameters
X_ols = SimulateModel2(t[overshoot:], X_syn[overshoot, :], mp_est, model=SIR, realtime=True)

n = 2 #spacing between plotted data
#plot infected
plt.plot(t,X_syn[:,1],c = "b")
plt.scatter(t[overshoot::n],X_ols[::n,1], marker = '+', c = "b")
#plot parameters
plt.plot(t[::n],mp[::n,0],c = "tab:orange")
plt.scatter(t[::n],mp_est[::n,0],c="tab:orange")
plt.xlabel("days")

plt.legend(["Infected data",r"$\beta$ data","Infected estimated",r"$\beta$ estimate"])


