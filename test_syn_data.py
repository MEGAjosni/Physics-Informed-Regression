# Set cwd to file dirand import relevant packages
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

from models import SIR, S3I3R, TripleRegionSIR, MultivariantSIR
from sim_functions import SimulateModel, LeastSquareModel, NoneNegativeLeastSquares
from tikzplotlib import save
import numpy as np
import matplotlib.pyplot as plt

# %%
'''
################################################
###### >>>>> The standard SIR model <<<<< ######
###############################################
'''
# ----------------------------------------------------------------------------------------------------
x0 = [1-0.0001, 0.0001, 0.0]          # Initial state.
mp = [0.5, 1/3]                     # Model parameters.
T_sim = 80                         # Number of time steps that should be simulated.
T_train = 50                        # Model trained on the first T_train time steps.
# ----------------------------------------------------------------------------------------------------

# Generate synthetic data
t = np.arange(T_sim)
X = SimulateModel(t, x0, mp, model=SIR)

# Estimate parameters and project
mp_est = LeastSquareModel(t[:T_train], X[0:T_train, :], model=SIR, normalize=False)
X_est = SimulateModel(t, X[0, :], mp_est, model=SIR)

# Plot results
l_width = 4.0
plt.plot(t[T_train:], X[T_train:,:], 'k-',linewidth=l_width)#training data
plt.plot(t[:T_train+1], X[:T_train+1,:], '-',linewidth=l_width)#true data
plt.plot(t, X_est, '--',linewidth=l_width+1.0)#predicted data
plt.legend([r"$S_{true}$","$I_{true}$","$R_{true}$","$S_{train}$","$I_{train}$","$R_{train}$","$S_{pred}$","$I_{pred}$","$R_{pred}$"])
plt.xlabel("Time [days]")
plt.ylabel("Population fraction")
save("PIR_SIR_synt_const_states.tex")
plt.show()

# %%
'''
#######################################################
###### >>>>> The 7 compartment S3I3R model <<<<< ######
#######################################################
'''

# ----------------------------------------------------------------------------------------------------
x0 = [0.9999,0.0001,0.0,0.0,0.0,0.0,0.0]         # Initial state.
beta = 0.5
gamma1 = 1/3 
gamma2 = 1/20
gamma3 = 1/20
phi1 = 1/20
phi2 = 1/20
theta = 1/10
tau = 0
mp = [beta, gamma1, gamma2, gamma3, phi1, phi2, theta, tau]
#mp = [0.2, 1/7, 1/14, 1/9, 0.005, 0.05, 0.2, 0]     # Model parameters.
T_sim = 7*6                                         # Number of time steps that should be simulated.
T_train = 7*4                                        # Model trained on the first T_train time steps.
# ----------------------------------------------------------------------------------------------------

# Generate synthetic data
t = np.arange(T_sim)
X = SimulateModel(t, x0, mp, model=S3I3R)

# Estimate parameters and project
mp_est = LeastSquareModel(t[:T_train], X[0:T_train, :], model=S3I3R, normalize=False, fix_params=[[1, gamma1], [2, gamma2],[3,gamma3]])
X_est = SimulateModel(t, X[0, :], mp_est, model=S3I3R)

# Plot results
l_width = 4.0
plt.plot(t[T_train:], X[T_train:,2:5], 'k-',linewidth=l_width)#training data
plt.plot(t[:T_train+1], X[:T_train+1,2:5], '-',linewidth=l_width)#true data
plt.plot(t, X_est[:,:2:5], '--',linewidth=l_width+1.0)#predicted data
plt.legend([r"$S_{true}$","$I_{true}$","$R_{true}$","$S_{train}$","$I_{train}$","$R_{train}$","$S_{pred}$","$I_{pred}$","$R_{pred}$"])
plt.xlabel("Time [days]")
plt.ylabel("Population fraction")
save("PIR_S3I3R_synt_const.tex")
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
