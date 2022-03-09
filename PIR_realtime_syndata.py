# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 17:50:21 2021

@author: Marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import paramest_functions_OLS as pest_OLS
import scipy.integrate
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel2, LeastSquareModel, NoneNegativeLeastSquares
from tikzplotlib import save

#%% SIR model
#initialise duration of simulation
simdays = 70
t1 = 0
t2 = t1+simdays


############ Real time estimations ################################
over_time = True #set "True" if estimations are needed in real time
overshoot = 1 #the amount of previous days included for parameter estimations in real time


#initialize IC and parameters for the syntetic data
X0 = [1-0.005, 0.005, 0.00000000e+00] #initial conditions
beta= 0.4  #rate of transmission
gamma = 1/3     #rate of recovery
noise_var = 0 #variance of added noise

t = np.arange(simdays)

############ Which parameters to find ####################################
pass_beta = None #"None" if you need a beta estimation, otherwise put a "beta" value
pass_gamma = None #"None" if you need a gamma estimation, otherwise put a "gamma" value


#### plots ##############
include_params = True #include parameters in plot



#Customize function for beta
def betacurve1(t,beta):
    return 0.05*np.sin(2*np.pi/simdays*t)+beta

def betacurve2(t,beta):
    if t < simdays/2:
        return 2*beta/simdays*t
    else:
        return beta
    
def betacurve3(t,beta):
    if t < simdays/2:
        return beta/2
    else:
        return beta*3/2

#Generate synthetic data
mp = np.array([[betacurve3(i,beta),gamma] for i in range(simdays)])


X_syn = SimulateModel2(t, X0, mp, model=SIR, realtime=True)   

    #real time parameters using OLS
mp_est = pest_OLS.SIR_params_over_time_OLS(
        t1 = t1,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn,
        beta = pass_beta,
        gamma = pass_gamma)
#real time paramaters using PINNS
mp_pinn = np.zeros((simdays-1,2))

fname = "pinn_params_SIR2"
file = open(fname+".out",'r')
k = 0
for line in file.readlines():
    mp_pinn[k] = line.split(",")
    k+= 1

#simulation using retrieved parameters
X_ols = SimulateModel2(t[overshoot:], X_syn[overshoot, :], mp_est, model=SIR, realtime=over_time)
X_pinn = SimulateModel2(t[overshoot:], X_syn[overshoot, :], mp_pinn, model=SIR, realtime=over_time)


n=1
plt.plot(t[::n],mp_est[::n,:],'o',label = [r"$\hat{\beta}$","$\hat{\gamma}$"])
#plt.plot(t[:-1],mp_pinn,'o',)
plt.plot(t,mp,label = [r"$\beta$","$\gamma$"])
plt.legend()
plt.ylim(0,1)
#save("PIR_SIR_synt_vary_params.tex")
plt.show()

#%% S3I3R

simdays = 28*2
t1 = 0
t2 = t1+simdays
overshoot = 1

beta = 0.5
gamma1 = 1/3 
gamma2 = 1/20
gamma3 = 1/20
phi1 = 1/20
phi2 = 1/20
theta = 1/10
tau = 0.001
X0 = [0.99, 0.001, 0,0,0,0,0]

def beta_sincurve(t,beta):
    return 0.05*np.sin(2*np.pi/simdays*t)+beta
t= np.arange(0, simdays)

mp  = np.vstack([0.05*np.sin(2*np.pi/simdays*t)+beta,
                  gamma1*np.ones(len(t)),
                  gamma2*np.ones(len(t)),
                  gamma3*np.ones(len(t)),
                  phi1*np.ones(len(t)),
                  phi2*np.ones(len(t)),
                  theta*np.ones(len(t)),
                  tau*np.ones(len(t))]).T



X_syn = SimulateModel2(t, X0, mp, model=S3I3R, realtime=True)

mp_est = pest_OLS.params_over_time_expanded_LA(
        t1 = t1 ,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn,
        mp = np.array([gamma1,gamma2,gamma3])
        )

mp_pinn = np.zeros((simdays-1,4))
fname = "varying_parameters_S3I3R_v5"
file = open(fname+".out",'r')
k = 0
for line in file.readlines():
    mp_pinn[k] = line.split(",")
    k+= 1


plt.plot(np.arange(simdays+1),mp_est.T,'o',label = [r"$\hat{\beta}$","$\hat{\phi}_1$","$\hat{\phi}_2$","$\hat{\theta}$"])
plt.plot(np.arange(simdays),[[beta_sincurve(i,beta),phi1,phi2,theta] for i in range(simdays)],'-', label=[r"$\beta$","$\phi_1$","$\phi_2$","$\theta$"])
plt.legend()
save("PIR_S3I3R_synt_vary_params.tex")
plt.show()

