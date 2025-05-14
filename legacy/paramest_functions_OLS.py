# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 21:01:02 2021

@author: Marcus
"""

import statistics
import numpy as np

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
        X = (X-X.mean(axis=0))/X.std(axis=0)
    #dX = X[1:,:]-X[:-1,:]
    dX = np.gradient(X,axis=0)
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


def estimate_params_expanded_LA(
        t1: int,  # Start
        t2: int,  # Stop
        X: np.array,
        mp: np.array,  # Known model parameters [gamma1, gamma2, gamma3]
):
    X_0 = X[t1,:]
    N = sum(X_0)
    simdays = t2 - t1

    dX = np.gradient(X,axis=0)
    dX_norms = np.linalg.norm(dX, axis=0)

    for k in range(t1,t2):
        dxt = dX[k,:]
        xt = X[k,:]
        # xt = [S I1 I2 I3 R1 R2 R3]

        A = np.array(
            [
                [-xt[1]*xt[0]/N,    0,      0,      0],
                [xt[0]*xt[1]/N,     -xt[1], 0,      0],
                [0,                 xt[1],  -xt[2], 0],
                [0,                 0,      xt[2],  -xt[3]],
                [0,                 0,      0,      xt[3]]
            ]
        )

        b = np.array(
            [
                dxt[0] + dxt[5]*xt[0]/(xt[0]+xt[1]+xt[4]),
                dxt[1] + xt[1]*mp[0] + xt[1]*dxt[5]/(xt[0]+xt[1]+xt[4]),
                dxt[2] + mp[1]*xt[2],
                dxt[3] + mp[2]*xt[3],
                dxt[6]
            ]
        )

        if k == t1:
            As = A
            bs = b
        else:
            As = np.concatenate([As,A],axis = 0)
            bs = np.concatenate([bs,b])

    beta,phi1,phi2,theta = np.linalg.lstsq(As, bs, rcond=None)[0]
    mp2 = [beta, phi1, phi2, theta]
    return np.array(mp2)

def params_over_time_expanded_LA(
        t1: int,
        t2: int,
        overshoot: int,
        X: np.array,
        mp: np.array
):

    simdays = (t2 - t1) + 1
    params = np.zeros((4, simdays))

    for k in range(simdays):
        params[:, k] = estimate_params_expanded_LA(
            t1=t1 + k - overshoot,
            t2=t1 + k,
            X=X,
            mp=mp,
        )
        # print(params[:, 0:k+1])

    return params