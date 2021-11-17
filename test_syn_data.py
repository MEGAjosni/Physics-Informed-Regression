# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 00:08:10 2021

@author: jonas
"""
''
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from models import SIR, S3I3R
from sim_functions import SimulateModel, LeastSquareModel
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib

# >>> SIR <<<
'''
errs = []

for i in range(1,101):
    for j in range(1,101):
        print(i)
        
        beta = 0.5*i/100
        gamma = 0.3*j/100
        
        #beta = 0.0+0.5*np.random.random()
        #gamma = 0.0+0.3*np.random.random()
    
        x0 = [5600000, 100000, 0]
        mp = [beta, gamma]
        
        t = np.arange(1, 100)
        
        X = SimulateModel(t, x0, mp, model=SIR)
        
        mp_est = LeastSquareModel(t, X, model=SIR, normalize=True)
        
        rel_err = abs((mp-mp_est)/mp)
        
        errs.append(rel_err)

errs = np.array(errs)

plt.hist(100*errs[:, 0], bins = 100)
plt.xlabel('Relative error of beta in %, simulated range [0.0, 0.5]')
tikzplotlib.save("graphix/betanonorm.tex")
plt.show()

plt.hist(100*errs[:, 1], bins = 100)
plt.xlabel('Relative error of gamma in %, simulated range [0.0, 0.3]')
tikzplotlib.save("graphix/gammanonorm.tex")
plt.show()

'''

x0 = [5600000, 100000, 0]
mp = [0.3, 1/50]

t = np.arange(100)

X = SimulateModel(t, x0, mp)

mp_est = LeastSquareModel(t[:14], X[0:14, :])

X_est = SimulateModel(t[14:], X[14, :], mp_est)

plt.plot(t, X)
plt.plot(t[14:], X_est, '--')
plt.show()

# >>> S3I3R <<<
'''
errs = []

for i in range(10000):
    print(i)

    x0 = [5600000, 100000, 1000, 10, 0, 0, 0]
    # mp = [0.6, 1/7, 1/14, 1/9, 0.005, 0.05, 0.2, 0]
    

    beta = 0.0+0.5*np.random.random()
    gamma1 = 0.0+0.3*np.random.random()
    gamma2 = 0.0+0.3*np.random.random()
    gamma3 = 0.0+0.3*np.random.random()
    phi1 = 0.0+0.03*np.random.random()
    phi2 = 0.0+0.3*np.random.random()
    theta = 0.0+0.5*np.random.random()

    mp = [beta, gamma1, gamma2, gamma3, phi1, phi2, theta, 0]

    t = np.arange(1, 100)
    
    X = SimulateModel(t, x0, mp, model=S3I3R)
    
    mp_est = LeastSquareModel(t, X, model=S3I3R, normalize=False, fix_params=[None, None, None, None, None, None, None, 0])
    
    # X_est = SimulateModel(t, x0, np.append(mp_est, 0), model=S3I3R)
    
    errs.append(np.mean((mp[:7]-mp_est)/mp[:7]))

plt.hist(100*errs, bins = 100)
plt.xlabel('Relative error in %')
tikzplotlib.save("graphix/S3I3R.tex")
plt.show()
'''
