# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 22:26:07 2021

@author: jonas
"""

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


data, date_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")
country = GetCountryCode('Denmark')
dataDK = ExtractContries(data, country)
SIRdata = SIRdataframe(dataDK, dark_number_scalar = 1, standardize=False)

markday = '2020-08-14'
mark = date_dict[markday]
interval = 7
project = 14
sample_days = 14
gamma = 1/7

betas = np.array([])

ts = np.arange(mark, mark+project)

for i in range(sample_days):
    t = np.arange(mark-interval-i, mark-i)
    beta = LeastSquareModel(t, SIRdata.iloc[t], model=SIR, fix_params=([None, gamma]), normalize=True)
    betas = np.append(betas, beta)
    
mu = np.mean(betas)
sigma = np.std(betas)

SIRupper = SimulateModel(ts, SIRdata.iloc[mark], [mu - 3*sigma, gamma])
SIRlower = SimulateModel(ts, SIRdata.iloc[mark], [mu + 3*sigma, gamma])

plt.plot(ts, SIRlower[:, 1], 'g--')
plt.plot(ts, SIRupper[:, 1], 'g--', label='_nolegend_')
# plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='green')

SIRupper = SimulateModel(ts, SIRdata.iloc[mark], [mu - 2*sigma, gamma])
SIRlower = SimulateModel(ts, SIRdata.iloc[mark], [mu + 2*sigma, gamma])

plt.plot(ts, SIRlower[:, 1], 'b--')
plt.plot(ts, SIRupper[:, 1], 'b--', label='_nolegend_')
# plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='blue', alpha=0.5)

SIRupper = SimulateModel(ts, SIRdata.iloc[mark], [mu - 1*sigma, gamma])
SIRlower = SimulateModel(ts, SIRdata.iloc[mark], [mu + 1*sigma, gamma])

plt.plot(ts, SIRlower[:, 1], 'm--')
plt.plot(ts, SIRupper[:, 1], 'm--', label='_nolegend_')
# plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='blue', alpha=0.5)

plt.plot(ts, SIRdata.iloc[ts]['I'], 'rx')
plt.legend(['99.7% Confidence Interval','95% Confidence Interval','68.2% Confidence Interval','Observed Infected'])
plt.title('Projecting ' + str(project) + ' days forward from ' + markday)
tikzplotlib.save("graphix/20200814.tex")

plt.show()