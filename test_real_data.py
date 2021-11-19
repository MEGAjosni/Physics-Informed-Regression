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
from models import TripleRegionSIR

# Load data
scandinavia = ['Denmark', 'Sweden', 'Norway']
data, date_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")
countries = [GetCountryCode(S) for S in scandinavia]
dataScan = ExtractContries(data, countries)

SIRdk = SIRdataframe(dataScan[countries[0]], N = 5800000, dark_number_scalar = 1, standardize=False)
SIRse = SIRdataframe(dataScan[countries[1]], N = 10400000, dark_number_scalar = 1, standardize=False)
SIRno = SIRdataframe(dataScan[countries[2]], N = 5400000, dark_number_scalar = 1, standardize=False)

# %%
'''
###############################################
###### >>>>> Estimating parameters <<<<< ######
###############################################
'''

# ----------------------------------------------------------------------------------------------------
t0 = '2021-01-01'           # Day the from which the projection is made
T_train = 14                 # Number of days used to estimate beta
T_proj = 14                 # Number of days to project forward from t0.
# ----------------------------------------------------------------------------------------------------

# Prepare data for model
t0 = date_dict[t0]
t = np.arange(t0-T_train, t0+T_proj)
X = np.concatenate([SIRdk.iloc[t], SIRse.iloc[t], SIRno.iloc[t]], axis=1)

# Estimate parameters
mp_est = LeastSquareModel(t[:T_train], X[:T_train, :], model=TripleRegionSIR, normalize=True)
X_est = SimulateModel(t[T_train:], X[T_train, :], mp_est, model=TripleRegionSIR)


# Plot results
plt.plot(t, X[:, 1::3])
plt.plot(t[T_train:], X_est[:, 1::3], '--')
plt.legend(['Observed Infected DK', 'Observed Infected SE', 'Observed Infected NO', 'Predicted Infected DK', 'Predicted Infected SE', 'Predicted Infected NO'])
plt.show()