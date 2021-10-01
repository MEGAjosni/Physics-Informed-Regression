# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 21:01:02 2021

@author: Marcus
"""

import statistics
import numpy as np
import tqdm

def SIR_params_OLS(
        t1: int,  # Start: index for first entry included from X
        t2: int,  # Stop: index for last entry included from X
        X: np.array, #data: n by 3 matrix, with column entries [S I R] and row entries correspondign to a day t
        beta = None,
        gamma = None, #if None then gamma will be computed using OLS. 
                      #Otherwise only beta will be estimated, using provided gamma
        normalised = False,
):
    # *** Description ***
    # Computes the parameters for the basic SIR model using the OLS-method

    # *** Output ***
    # Optimal parameters [beta,gamma]
    
    
    
    if normalised:
        X = (X-X.mean(axis=0))/X.var(axis=0)
    dX = X[1:,:]-X[:-1,:]
    N = sum(X[0,:])

    
    for t in range(t1,t2):
        dS, dI, dR = dX[t,:]
        S, I, R = X[t,:]
        
        b = dX[t,:]
        A1 = np.array([-S*I/N, S*I/N,0])
        A2 = np.array([0,-I,I])
        
        if beta is None:
            A = A1.reshape(3,1)
        else:
            b -= beta*A1
        
        if gamma is None:
            A = A2.reshape(3,1)
        else:
            b -= gamma*A2
        if gamma is None and beta is None:
            A = np.hstack((A1.reshape(3,1),A2.reshape(3,1)))
    
        if t == t1:
            As = A
            bs = b
        else:
            As = np.concatenate([As,A],axis = 0)
            bs = np.concatenate([bs,b])

    params = np.linalg.lstsq(As, bs, rcond=None)[0]
    return params

def SIR_params_over_time_OLS(
        t1: int,
        t2: int,
        overshoot: int,
        X: np.array,
        beta = None,
        gamma = None,
):    
    # *** Description ***
    # Computes the parameters for the basic SIR model using the OLS-method.
    # in real time.

    # *** Output ***
    # the optimal parameters [beta,gamma] for all n=t2-t1 days,
    # each row corresponds to the optimal parameters for a single day
    simdays = t2-t1
    
    m = int(beta is None)+int(gamma is None)
    mps= np.zeros((simdays,m))

    for k in range(t1,t2): 
        if k - overshoot+1 < 0:
            starttime = 0
        else:
            starttime = k - overshoot+1
            
        mps[k-t1,:] = SIR_params_OLS(
            t1 = starttime,
            t2 = k+1,
            X = X,
            beta = beta,
            gamma = gamma,
        )
        
    if beta is None and gamma is None:
        return mps
    elif beta is None:
        gammas = gamma*np.ones((len(mps),1))
        return np.hstack((mps,gammas))
    elif gamma is None:
        betas = beta*np.ones((len(mps),1))
        return np.hstack((betas,mps))
    return mps