# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 00:17:49 2021

@author: Marcus
"""

import matplotlib.pyplot as plt
import numpy as np
import paramest_functions_OLS as pest_OLS
import scipy.integrate
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel2, LeastSquareModel, NoneNegativeLeastSquares



#initialise duration of simulation
simdays = 100
t1 = 0
t2 = t1+simdays


############ Real time estimations ################################
over_time = True #set "True" if estimations are needed in real time
overshoot = 4 #the amount of previous days included for parameter estimations in real time


#initialize IC and parameters for the syntetic data
X0 = [1-0.005, 0.005, 0.0] #initial conditions
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

#mp = np.array([[0.05*np.sin(4*np.pi/simdays*i)+beta,gamma] for i in range(simdays)])

#X_syn = SimulateModel2(t, X0, mp, model=SIR, realtime=over_time)

from data_functions import ReadDataFromCsvFile, ExtractContries, SIRdataframe, getPopulation, GetCountryCode, DownloadData


data, date_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")
country = GetCountryCode('Denmark')
dataDK = ExtractContries(data, country)
SIRdata = SIRdataframe(dataDK, N = 5800000, dark_number_scalar = 1, standardize=False)
X_syn = SIRdata.to_numpy()

day0 = 619
X_syn = X_syn[day0:day0+simdays]


    #real time parameters using OLS
mp_est = pest_OLS.SIR_params_over_time_OLS(
        t1 = t1,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn,
        beta = pass_beta,
        gamma = pass_gamma)
#real time paramaters usinxg PINNS
mp_pinn = np.zeros((simdays-1,2))

fname = "pinn_params_SIR"
file = open(fname+".out",'r')
k = 0
for line in file.readlines():
    mp_pinn[k] = line.split(",")
    k+= 1

#simulation using retrieved parameters
X_ols = SimulateModel2(t[overshoot:], X_syn[overshoot, :], mp_est, model=SIR, realtime=over_time)
X_pinn = SimulateModel2(t[overshoot:], X_syn[overshoot, :], mp_pinn, model=SIR, realtime=over_time)


# monte carlo
T_train = 7               # Number of days used to estimate beta.
T_proj = 100                 # Number of days to project forward from t0.
T_samp = 7                  # Number of proceeding timesteps of t0 for which beta should be sampled.
gamma = 1/9                 # Gamma is assumed constant.
n_mc = 1000
         # Number of random samples to make.:
             
betas = mp_est[simdays-T_samp:,0]
mu = np.mean(betas)
sigma = np.std(betas)

# Pick out n_mc random samples 
mc_betas = np.random.normal(mu, sigma, n_mc)

# Simulate each sample
ts = np.arange(simdays, simdays + T_proj + 1)
mc_sims = np.empty((T_proj + 1, n_mc))

for idx, beta in enumerate(mc_betas):
    mc_sims[:, idx] = SimulateModel2(ts, X_syn[simdays-1,:], [beta, gamma], realtime = False)[:, 1]

mu_sim = np.mean(mc_sims, axis=1)
std_sim = np.std(mc_sims, axis=1)
t0  =simdays

CI_colors = ['darkslateblue', 'slateblue', 'mediumslateblue']
CI_style = '--'

#plots
n = 1
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(t,X_syn[:,1])
ax.scatter(t[overshoot::n],X_ols[::n,1], marker = '+', c = "b")

for idx, color in reversed(list(enumerate(CI_colors, 1))):
    ax.fill_between(ts, mu_sim - idx * std_sim, mu_sim + idx * std_sim, color=color, alpha=1)
#if include_params:
ax.legend(['I_data','I_OLS,', '99.7% Confidence Interval', '95% Confidence Interval', '68.2% Confidence Interval'],loc="upper left")
ax2.scatter(t[::],mp_est[::,0], c = "r")

ax2.set_ylim(0.05,0.3)
ax2.legend(["b OLS","g OLS","b true", "g true"],loc = "upper right")


