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

markday = '2021-05-14'
mark = date_dict[markday]
interval = 7

betas = np.array([])

ts = np.arange(mark, mark+7)

for i in range(14):
    t = np.arange(mark-interval-i, mark-i)
    beta = LeastSquareModel(t, SIRdata.iloc[t], model=SIR, fix_params=([0, 1/7]))
    betas = np.append(betas, beta)
    
mu = np.mean(betas)
sigma = np.std(betas)

SIRupper = SimulateModel(ts, SIRdata.iloc[mark], [mu - 3*sigma, 1/7])
SIRlower = SimulateModel(ts, SIRdata.iloc[mark], [mu + 3*sigma, 1/7])

plt.plot(ts, SIRlower[:, 1], 'g--')
plt.plot(ts, SIRupper[:, 1], 'g--', label='_nolegend_')
# plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='green')

SIRupper = SimulateModel(ts, SIRdata.iloc[mark], [mu - 2*sigma, 1/7])
SIRlower = SimulateModel(ts, SIRdata.iloc[mark], [mu + 2*sigma, 1/7])

plt.plot(ts, SIRlower[:, 1], 'b--')
plt.plot(ts, SIRupper[:, 1], 'b--', label='_nolegend_')
# plt.fill_between(ts, SIRlower[:, 1], SIRupper[:, 1], color='blue', alpha=0.5)

plt.plot(ts, SIRdata.iloc[ts]['I'], 'rx')
plt.legend(['99.7% Confidence Interval','95% Confidence Interval','Observed Infected'])
plt.title('Projecting ' + str(interval) + ' days forward from ' + markday)
tikzplotlib.save("graphix/test.tex")

plt.show()