# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 00:17:49 2021

@author: Marcus
"""

import basic_ivp_funcs as b_ivp
import matplotlib.pyplot as plt
import numpy as np
from data_functions import *
import get_synth_data as gsd
import paramest_functions_OLS as pest_OLS
import pandas as pd
import datetime as dt

#initialise duration of simulation
simdays = 100
t1 = 0
t2 = t1+simdays


############ Real time estimations ################################
over_time = True #set "True" if estimations are needed in real time
overshoot = 14 #the amount of previous days included for parameter estimations in real time


#initialize IC and parameters for the syntetic data
X_0 = [10000, 30, 0] #initial conditions
beta= 0.23  #rate of transmission
gamma = 0.08 #rate of recovery
noise_var = 0 #variance of added noise
stepsize = 0.1 #stepsize in Runge-Kutta iteration

T = np.arange(simdays)

############ Which parameters to find ####################################
pass_beta = None #"None" if you need a beta estimation, otherwise put a "beta" value
pass_gamma = None #"None" if you need a gamma estimation, otherwise put a "gamma" value


#### plots ##############
include_params = True #include parameters in plot



#Generate synthetic data or use real data
mp = [beta, gamma]

"""
t_syn, X_syn = gsd.Create_synth_data(X_0 = X_0,
                      mp = mp,
                      model_type = 'basic',
                      simdays = simdays+overshoot,
                      stepsize =  stepsize,
                      noise_var = noise_var,
                      )
"""

    #real time parameter estimation from data
mp_est = pest_OLS.SIR_params_over_time_OLS(
        t1 = t1,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn,
        beta = pass_beta,
        gamma = pass_gamma)


#simulation using retrieved parameters
t_sim, SIR_sim = b_ivp.simulateSIR(
    X_0=X_0,
    mp=mp_est,
    simtime=simdays,
    stepsize = stepsize,
    method=b_ivp.RK4)


#plots
n = 5
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.scatter(t_syn[::n*10],X_syn[::n,0])
#ax.scatter(T[::n],X_syn[::n,1])
#ax.scatter(T[::n],X_syn[::n,2])
ax.plot(t_sim,SIR_sim[:,0])
#ax.plot(t_sim,SIR_sim[:,1])
#ax.plot(t_sim,SIR_sim[:,2])
#ax.legend(["S","I","R"],loc="upper left")
ax.legend("S",loc="upper left")

if include_params:
    ax2.plot(mp_est[:,0], c = "r")
    ax2.plot(mp_est[:,1], c = "tab:purple")
    if syn:
        ax2.plot(T,beta*np.ones(len(T)),"--",c = "r")
        ax2.plot(T,gamma*np.ones(len(T)),"--",c = "tab:purple")
    ax2.set_ylim(0,0.30)
    ax2.legend(["b est","g est","b true", "g true"],loc = "upper right")

#%% Once again with real data

start_date = "2020-12-01"
#initilize SIR data
simdays = 100
t1 = 0
t2 = t1+simdays

data, attribute_dict, date_dict, country_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")
X = ExtractContries(data,"Denmark",country_dict)
X = SIRdataframe(X,gamma,dark_number_scalar=1).to_numpy()
X = X[date_dict[start_date]:date_dict[start_date]+simdays+overshoot]

