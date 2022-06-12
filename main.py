
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
from data_functions import ReadDataFromCsvFile, ExtractContries, SIRdataframe, getPopulation, GetCountryCode, DownloadData



#initialise duration of simulation
simdays = 70
t1 = 0
t2 = t1+simdays
overshoot = 1  # all the data from index i-overshoot to i will be used to estimate parameters at day i
t = np.arange(simdays)

#Choose country
country = 'Denmark'
N = 5800000 #population at date 2020-01-03

# get WHO data
data, date_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")
country = GetCountryCode(country)
dataDK = ExtractContries(data, country)
SIRdata = SIRdataframe(dataDK, N = N, dark_number_scalar = 1, standardize=False)
X_syn = SIRdata.to_numpy()

#choose start day. X_syn[0] corresponds to the state in denmark at 2020-01-03
day0 = 600
X_syn = X_syn[day0:day0+simdays]


    #real time parameters using OLS
mp_est = pest_OLS.SIR_params_over_time_OLS(
        t1 = t1,
        t2 = t2,
        overshoot = overshoot,
        X = X_syn)

#simulation using retrieved parameters
X_ols = SimulateModel2(t[overshoot:], X_syn[overshoot, :], mp_est, model=SIR, realtime=True)


#plots
n = 1
#S
#plt.scatter(t[overshoot::n],X_ols[::n,0],label = '$S_{est}$')
#plt.plot(t,X_syn[:,0],label = '$S_{data}$')
#I
plt.scatter(t[overshoot::n],X_ols[::n,1], label = '$I_{est}$')
plt.plot(t,X_syn[:,1],label = '$I_{data}$')
#R
#plt.scatter(t[overshoot::n],X_ols[::n,2],label = '$R_{est}$')
#plt.plot(t,X_syn[:,2],label = '$R_{data}$')
plt.ylabel('Persons/Population')
plt.xlabel('Time [days]')
plt.legend()
plt.show()

