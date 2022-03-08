
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 00:17:49 2021

@author: Marcus
"""

import matplotlib.pyplot as plt
import numpy as np
import get_synth_data as gsd
import tikzplotlib
import paramest_functions_OLS as pest_OLS
import scipy.integrate
from models import SIR, S3I3R, TripleRegionSIR
from sim_functions import SimulateModel2, LeastSquareModel, NoneNegativeLeastSquares



#initialise duration of simulation
simdays = 70
t1 = 0
t2 = t1+simdays


############ Real time estimations ################################
over_time = True #set "True" if estimations are needed in real time
overshoot = 4 #the amount of previous days included for parameter estimations in real time


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
#real time paramaters using PINNS
mp_pinn = np.zeros((simdays-1,2))

fname = "pinn_params_SIR3"
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
#S
plt.scatter(t[overshoot::n],X_ols[::n,0])
plt.plot(t,X_syn[:,0])
#I
plt.scatter(t[overshoot::n],X_ols[::n,1])
plt.plot(t,X_syn[:,1])
#R
plt.scatter(t[overshoot::n],X_ols[::n,2])
plt.plot(t,X_syn[:,2])
plt.ylabel('Persons/Population')
plt.xlabel('Time [days]')
plt.legend(['$S_{data}$','$I_{data}$','$R_{data}$','$S_{est}$','$I_{est}$','$R_{est}$'])
tikzplotlib
plt.show()

#parameter plot
if over_time:
    plt.plot(t,mp[:,0],c = "r")
    plt.plot(t,mp[:,1],c = "tab:purple")
else:
    plt.plot(t,beta*np.ones(len(t)),"--",c = "r")
    plt.plot(t,gamma*np.ones(len(t)),"--",c = "tab:purple")
plt.plot(t,mp_pinn[:,0],c = "r",marker ='o',linestyle='None')
plt.plot(t,mp_pinn[:,1],c = "tab:purple",marker = 'o', linestyle='None')
plt.ylim((0.2,0.6))
plt.xlabel('Time [days]')
plt.legend([r'$\beta_{data}$',r'$\gamma_{data}$',r'$\beta_{est}$',r'$\gamma_{est}$'])
plt.show()
