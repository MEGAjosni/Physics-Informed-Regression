# Set cwd to file dirand import relevant packages
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from models import SIR, S3I3R, TripleRegionSIR, MultivariantSIR
from sim_functions import SimulateModel, LeastSquareModel, NoneNegativeLeastSquares
import numpy as np
import matplotlib.pyplot as plt

# %%
'''
################################################
###### >>>>> The standard SIR model <<<<< ######
################################################
'''
# ----------------------------------------------------------------------------------------------------
x0 = [5000000, 600000, 0]           # Initial state.
mp = [0.2, 1/9]                     # Model parameters.
T_sim = 100                         # Number of time steps that should be simulated.
T_train = 14                        # Model trained on the first T_train time steps.
# ----------------------------------------------------------------------------------------------------

# Generate synthetic data
t = np.arange(T_sim)
X = SimulateModel(t, x0, mp, model=SIR)


# Estimate parameters and project
mp_est = LeastSquareModel(t[:T_train], X[0:T_train, :], model=SIR, normalize=True)
X_est = SimulateModel(t[T_train:], X[T_train, :], mp_est, model=SIR)

# Plot results
plt.plot(t, X[:, 1], 'm.')
plt.plot(t[T_train:], X_est[:, 1], 'c-')
plt.legend(['Simulated Infected', 'Predicted Infected'])
plt.show()

# %%
'''
#######################################################
###### >>>>> The 7 compartment S3I3R model <<<<< ######
#######################################################
'''
# ----------------------------------------------------------------------------------------------------
x0 = [5600000, 100000, 1000, 10, 0, 0, 0]           # Initial state.
mp = [0.2, 1/7, 1/14, 1/9, 0.005, 0.05, 0.2, 0]     # Model parameters.
T_sim = 100                                         # Number of time steps that should be simulated.
T_train = 14                                        # Model trained on the first T_train time steps.
# ----------------------------------------------------------------------------------------------------

# Generate synthetic data
t = np.arange(T_sim)
X = SimulateModel(t, x0, mp, model=S3I3R)

# Estimate parameters and project
mp_est = LeastSquareModel(t[:T_train], X[0:T_train, :], model=S3I3R, normalize=True, fix_params=[[7, 0]])
X_est = SimulateModel(t[T_train:], X[T_train, :], mp_est, model=S3I3R)

# Plot results
plt.plot(t, X[:, 2:4])
plt.plot(t[T_train:], X_est[:, 2:4], '--')
plt.legend(['Simulated Hospitalized', 'Simulated ICU', 'Predicted Infected', 'Predicted ICU'])
plt.show()

# %%
'''
#################################################
###### >>>>> Scandinaivian SIR model <<<<< ######
#################################################

'''
# ----------------------------------------------------------------------------------------------------
x0 = [5700000, 100000, 0, 10000000, 300000, 0, 5300000, 50000, 0]   # Initial state.
mp = [0.2, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05, 0.2, 1/9]       # Model parameters.
T_sim = 100                                                         # Number of time steps that should be simulated.
T_train = 14                                                        # Model trained on the first T_train time steps.
# ----------------------------------------------------------------------------------------------------

# Generate synthetic data
t = np.arange(T_sim)
X = SimulateModel(t, x0, mp, model=TripleRegionSIR)

# Estimate parameters and project
mp_est = LeastSquareModel(t[:T_train], X[:T_train, :], model=TripleRegionSIR, normalize=True)
X_est = SimulateModel(t[T_train:], X[T_train, :], mp, model=TripleRegionSIR)

# Plot results
plt.plot(t, X[:, 1::3])
plt.plot(t[T_train:], X_est[:, 1::3], '--')
plt.legend(['Simulated Infected DK', 'Simulated Infected SE', 'Simulated Infected NO', 'Predicted Infected DK', 'Predicted Infected SE', 'Predicted Infected NO'])
plt.show()

# %%
'''
################################################
###### >>>>> Multivariant SIR model <<<<< ######
################################################
'''
# ----------------------------------------------------------------------------------------------------
x0 = [5000000, 10000, 100000, 100000, 0]                           # Initial state.
mp = [[0]*50+[0.3]*49, 0.1, 0.1, 1/9, 1/9, 1/9]                                 # Model parameters.
T_sim = 100                                                         # Number of time steps that should be simulated.
T_train = 14                                                        # Model trained on the first T_train time steps.
# ----------------------------------------------------------------------------------------------------

# Generate synthetic data
t = np.arange(T_sim)
X = SimulateModel(t, x0, mp, model=MultivariantSIR)

# Estimate parameters and project
mp_est = LeastSquareModel(t[:T_train], X[:T_train, :], model=MultivariantSIR, normalize=True)
X_est = SimulateModel(t[T_train:], X[T_train, :], mp_est, model=MultivariantSIR)

# Plot results
plt.plot(t, X[:, 1:4])
plt.plot(t[T_train:], X_est[:, 1:4], '--')
plt.legend(['Simulated variant 1', 'Simulated variant 2', 'Simulated variant 3', 'Predicted variant 1', 'Predicted variant 2', 'Predicted variant 3'])
plt.show()
