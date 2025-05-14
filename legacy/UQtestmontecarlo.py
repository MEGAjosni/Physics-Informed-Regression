# Set cwd to file dirand import relevant packages
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

from data_functions import ReadDataFromCsvFile, ExtractContries, SIRdataframe, getPopulation, GetCountryCode, DownloadData
from sim_functions import LeastSquareModel, SimulateModel
from models import SIR

# Load data
data, date_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")
country = GetCountryCode('Denmark')
dataDK = ExtractContries(data, country)
SIRdata = SIRdataframe(dataDK, N = 5800000, dark_number_scalar = 1, standardize=False, gamma = 1/9)

# %%
'''
#########################################################
###### >>>>> UQ - Monte Carlo Synthetic data <<<<< ######
#########################################################
'''

# ----------------------------------------------------------------------------------------------------
x0 = [5600000, 50000, 0]    # Initial state.
t0 = 21
T_train = 14                # Number of days used to estimate beta.
T_proj = 70                 # Number of days to project forward from t0.
T_samp = 7                  # Number of proceeding timesteps of t0 for which beta should be sampled.
beta = 0.2                  # Beta used for generating data.
gamma = 1/9                 # Gamma is assumed constant.
n_mc = 1000                 # Number of random samples to make.
# ----------------------------------------------------------------------------------------------------

X = SimulateModel(np.arange(100), x0, [beta, gamma])

X += np.random.normal(0, 10000, X.shape)

# Sample beta over the last T_samp timesteps to obtain its distribution.
betas = np.array([])
for i in range(T_samp):
    t = np.arange(t0-T_train-i, t0-i)
    beta = LeastSquareModel(t, X[t, :], model=SIR, fix_params=[[1, gamma]], normalize=True)[0]
    betas = np.append(betas, beta)
    
mu = np.mean(betas)
sigma = np.std(betas)

# Pick out n_mc random samples 
mc_betas = np.random.normal(mu, sigma, n_mc)

# Simulate each sample
ts = np.arange(t0, t0 + T_proj + 1)
mc_sims = np.empty((T_proj + 1, n_mc))

for idx, beta in enumerate(mc_betas):
    mc_sims[:, idx] = SimulateModel(ts, X[t0, :], [beta, gamma])[:, 1]

mu_sim = np.mean(mc_sims, axis=1)
std_sim = np.std(mc_sims, axis=1)

CI_colors = ['darkslateblue', 'slateblue', 'mediumslateblue']
CI_style = '--'

# >>>>> Plot Monte Carlo simulations <<<<<
plt.plot(X[:, 1], color='orange', marker='x')
plt.plot(ts, mc_sims, color='royalblue')
plt.legend(['Observed Infected', 'Monte Carlo simulations'])
plt.show()

# >>>>> Plot confidence intervals <<<<<
for idx, color in reversed(list(enumerate(CI_colors, 1))):
    plt.fill_between(ts, mu_sim - idx * std_sim, mu_sim + idx * std_sim, color=color, alpha=1)

# Plot observations
plt.plot(X[:, 1], color='orange', marker='x')
plt.legend(['Observed Infected', '99.7% Confidence Interval', '95% Confidence Interval', '68.2% Confidence Interval'])
plt.show()

# %%
'''
####################################################
###### >>>>> UQ - Monte Carlo Real Data <<<<< ######
####################################################
'''

# ----------------------------------------------------------------------------------------------------
startday = '2020-12-12'     # Day the from which the projection is made.
T_train = 7                 # Number of days used to estimate beta.
T_proj = 50                 # Number of days to project forward from t0.
T_samp = 7                  # Number of proceeding timesteps of t0 for which beta should be sampled.
gamma = 1/9                 # Gamma is assumed constant.
n_mc = 1000                 # Number of random samples to make.
# ----------------------------------------------------------------------------------------------------


# Sample beta over the last T_samp timesteps to obtain its distribution.
t0 = date_dict[startday]
betas = np.array([])
for i in range(T_samp):
    t = np.arange(t0-T_train-i, t0-i)
    beta = LeastSquareModel(t, SIRdata.iloc[t], model=SIR, fix_params=[[1, gamma]], normalize=True)[0]
    betas = np.append(betas, beta)
    
mu = np.mean(betas)
sigma = np.std(betas)

# Pick out n_mc random samples 
mc_betas = np.random.normal(mu, sigma, n_mc)

# Simulate each sample
ts = np.arange(t0, t0 + T_proj + 1)
mc_sims = np.empty((T_proj + 1, n_mc))

for idx, beta in enumerate(mc_betas):
    mc_sims[:, idx] = SimulateModel(ts, SIRdata.iloc[t0], [beta, gamma])[:, 1]

mu_sim = np.mean(mc_sims, axis=1)
std_sim = np.std(mc_sims, axis=1)

CI_colors = ['darkslateblue', 'slateblue', 'mediumslateblue']
CI_style = '--'

# >>>>> Plot Monte Carlo simulations <<<<<
plt.plot(np.arange(t0 - T_train, t0 + T_proj), SIRdata.iloc[np.arange(t0 - T_train, t0 + T_proj)]['I'], color='orange', marker='x')
plt.plot(ts, mc_sims, color='royalblue')
plt.legend(['Observed Infected', 'Monte Carlo simulations'])
plt.title('Plot of Monte Carlo simulations from {}'.format(startday))
plt.show()

# >>>>> Plot confidence intervals <<<<<
for idx, color in reversed(list(enumerate(CI_colors, 1))):
    plt.fill_between(ts, mu_sim - idx * std_sim, mu_sim + idx * std_sim, color=color, alpha=1)

# Plot observations
plt.plot(np.arange(t0 - T_train, t0 + T_proj), SIRdata.iloc[np.arange(t0 - T_train, t0 + T_proj)]['I'], color='orange', marker='x')
plt.legend(['Observed Infected', '99.7% Confidence Interval', '95% Confidence Interval', '68.2% Confidence Interval'])
plt.title('Confidence Intervals of Monte Carlo simulation from {}'.format(startday))
plt.show()
