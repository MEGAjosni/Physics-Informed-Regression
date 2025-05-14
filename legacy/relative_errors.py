# Set cwd to file dir and import relevant packages
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from sim_functions import LeastSquareModel, SimulateModel
from models import SIR, S3I3R

# %%
'''
##############################################################################
###### >>>>> SIR - Relative errors when testing on synthetic data <<<<< ######
##############################################################################
'''
# Generate the parameters
betas = np.arange(0.05, 0.51, 0.01)
gammas = np.arange(0.05, 0.31, 0.01)

# Initial state
t = np.arange(0, 100)
x0 = [5700000, 100000, 0]
rel_error = []

# Run simulations
for beta in betas:
    print(beta)
    for gamma in gammas:
        mp = np.array([beta, gamma])
        X = SimulateModel(t, x0, mp, model=SIR)
        mp_est = LeastSquareModel(t, X, model=SIR)
        rel_error.append(abs(1 - mp_est/mp))

rel_error = np.array(rel_error)

# Plot and save results
plt.hist(rel_error[:, 0], bins = 50)
tikzplotlib.save('beta_error.tex')
plt.show()

plt.hist(rel_error[:, 1], bins = 50)
tikzplotlib.save('gamma_error.tex')
plt.show()

# %%
'''
################################################################################
###### >>>>> S3I3R - Relative errors when testing on synthetic data <<<<< ######
################################################################################
'''

n_samples = 10000

# Initial state
t = np.arange(0, 100)
x0 = [5700000, 100000, 1000, 10, 0, 0, 0]
rel_error = []

for i in range(n_samples):
    print(i)
    # Generate the parameters
    beta = np.random.uniform(0, 0.5)
    gamma1 = np.random.uniform(0, 0.3)
    gamma2 = np.random.uniform(0, 0.3)
    gamma3 = np.random.uniform(0, 0.3)
    phi1 = np.random.uniform(0, 0.03)
    phi2 = np.random.uniform(0, 0.3)
    theta = np.random.uniform(0, 0.5)
    
    mp = np.array([beta, gamma1, gamma2, gamma3, phi1, phi2, theta, 0])

    X = SimulateModel(t, x0, mp, model=S3I3R)
    mp_est = LeastSquareModel(t, X, model=S3I3R, fix_params=[[7, 0]], normalize=True)
    rel_error.append(abs(1 - mp_est[:-1]/mp[:-1]))

rel_error = np.array(rel_error)
