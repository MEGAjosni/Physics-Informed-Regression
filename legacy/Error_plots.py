# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:34:12 2021

@author: alboa
"""

import numpy as np

## computes mean squared error for .out file (SIR and S3I3R) just change init func and settings

def mpfun_betagen(t,beta,simdays):
    return [0.05*np.sin(2*np.pi/simdays*t)+beta]

def compute_MSE(True_params,filename,simdays):
    f = open(filename)
    lines = f.readlines()
    SE_PINN = [0]*len(True_params)
    
    # collect sum of squared errors in SE_PINN
    for ind,line in enumerate(lines):
        split_line = [float(x) for x in line.split(",")] # current line is now [beta,phi1,phi2,theta]
        for SE_ind, param in enumerate(split_line):
            SE_PINN[SE_ind] += (param - True_params[SE_ind][ind])**2
    
    # divide py Nw
    return [x/(simdays-1) for x in SE_PINN]

def init_S3I3R():
    simdays = 56
    t = np.arange(1,simdays)
    phi1_true = np.ones(simdays-1) * 1/20
    phi2_true = np.ones(simdays-1) * 1/20
    theta_true = np.ones(simdays-1) * 1/10
    beta_true = mpfun_betagen(t,0.5,simdays)[0]
    return [beta_true,phi1_true,phi2_true,theta_true],simdays

def init_SIR():
    simdays = 70
    t = np.arange(1,simdays)
    gamma_true = np.ones(simdays-1) * 1/3
    beta_true = mpfun_betagen(t,0.4,simdays)[0]
    return [beta_true,gamma_true],simdays

    
## main ##

filename = 'pinn_params_SIR3.out'
true_params,simdays = init_SIR()
MSE_SIR = compute_MSE(true_params, filename, simdays)
print(f'Mean squared error for SIR params : \n {MSE_SIR}')



# S3I3R #
filename = 'varying_parameters_S3I3R_v5.out'
true_params,simdays = init_S3I3R()
MSE_S3I3R = compute_MSE(true_params, filename, simdays)
print(f'Mean squared error for S3I3R params : \n {MSE_S3I3R}')






