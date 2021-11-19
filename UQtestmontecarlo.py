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
###################################################
###### >>>>> UQ - Monte Carlo approach <<<<< ######
###################################################
'''

# ----------------------------------------------------------------------------------------------------
t0 = '2020-12-12'           # Day the from which the projection is made.
T_train = 14                # Number of days used to estimate beta.
T_proj = 14                 # Number of days to project forward from t0.
T_samp = 7                  # Number of proceeding timesteps of t0 for which beta should be sampled.
gamma = 1/9                 # Gamma is assumed constant.
# ----------------------------------------------------------------------------------------------------


# Sample beta over the last T_samp timesteps to obtain its distribution.
t0 = date_dict[t0]
betas = np.array([])
for i in range(T_samp):
    t = np.arange(t0-T_train-i, t0-i)
    beta = LeastSquareModel(t, SIRdata.iloc[t], model=SIR, fix_params=[None, gamma], normalize=True)
    betas = np.append(betas, beta)
    
mu = np.mean(betas)
sigma = np.std(betas)


# Simulate different confidence intervals from the distribution of beta.
ts = np.arange(t0, t0 + T_proj)

SIRupper = SimulateModel(ts, SIRdata.iloc[t0], [mu - 3*sigma, gamma])
SIRlower = SimulateModel(ts, SIRdata.iloc[t0], [mu + 3*sigma, gamma])

plt.plot(ts, SIRlower[:, 1], 'g--')
plt.plot(ts, SIRupper[:, 1], 'g--', label='_nolegend_')
plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='g')

SIRupper = SimulateModel(ts, SIRdata.iloc[t0], [mu - 2*sigma, gamma])
SIRlower = SimulateModel(ts, SIRdata.iloc[t0], [mu + 2*sigma, gamma])

plt.plot(ts, SIRlower[:, 1], 'b--')
plt.plot(ts, SIRupper[:, 1], 'b--', label='_nolegend_')
plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='b', alpha=0.5)

SIRupper = SimulateModel(ts, SIRdata.iloc[t0], [mu - 1*sigma, gamma])
SIRlower = SimulateModel(ts, SIRdata.iloc[t0], [mu + 1*sigma, gamma])

plt.plot(ts, SIRlower[:, 1], 'm--')
plt.plot(ts, SIRupper[:, 1], 'm--', label='_nolegend_')
plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='m', alpha=0.5)

plt.plot(ts, SIRdata.iloc[ts]['I'], color='orange', marker='x')
plt.legend(['99.7% Confidence Interval','95% Confidence Interval','68.2% Confidence Interval','Observed Infected'])
plt.title('Projecting ' + str(T_proj) + ' days forward from ' + [key for key, value in date_dict.items() if value == t0][0])
tikzplotlib.save("graphix/20200814.tex")

plt.show()