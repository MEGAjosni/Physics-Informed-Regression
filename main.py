# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 00:17:49 2021

@author: Marcus
"""

import matplotlib.pyplot as plt
import numpy as np
from data_functions import *
import get_synth_data as gsd
import paramest_functions_OLS as pest_OLS
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel, LeastSquareModel, NoneNegativeLeastSquares

#initialise duration of simulation
simdays = 100
t1 = 0
t2 = t1+simdays


############ Real time estimations ################################
over_time = True #set "True" if estimations are needed in real time
overshoot = 14 #the amount of previous days included for parameter estimations in real time


#initialize IC and parameters for the syntetic data
X0 = [10000, 30, 0] #initial conditions
beta= 0.23  #rate of transmission
gamma = 0.08 #rate of recovery
noise_var = 0 #variance of added noise
stepsize = 0.1 #stepsize in Runge-Kutta iteration

t = np.arange(simdays)

############ Which parameters to find ####################################
pass_beta = None #"None" if you need a beta estimation, otherwise put a "beta" value
pass_gamma = None #"None" if you need a gamma estimation, otherwise put a "gamma" value


#### plots ##############
include_params = True #include parameters in plot



#Generate synthetic data or use real data
#mp = [beta, gamma]
mp = [[0.05*np.sin(2*np.pi/simdays*i)+0.2,gamma] for i in range(simdays)]

X_syn = SimulateModel(t, X0, mp, model=SIR, realtime=over_time)


    #real time parameter estimation from data
mp_est = pest_OLS.SIR_params_over_time_OLS(
        t1 = t1,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn,
        beta = pass_beta,
        gamma = pass_gamma)


#simulation using retrieved parameters
X_est = SimulateModel(t[overshoot:], X_syn[overshoot, :], mp_est, model=SIR, realtime=over_time)


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
    ax2.plot(t,beta*np.ones(len(t)),"--",c = "r")
    ax2.plot(t,gamma*np.ones(len(t)),"--",c = "tab:purple")
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

