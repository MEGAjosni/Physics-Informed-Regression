# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:12:40 2021

@author: Marcu
"""

import get_data as gd
import datetime as dt
import pandas as pd
#import tikzplotlib
import numpy as np
## INITIALISE DATA

print(gd.infect_dict['Test_pos_over_time_antigen'])



DI1 = gd.infect_dict['Test_pos_over_time']['NewPositive'].to_numpy(copy=True)
DI2 = gd.infect_dict['Test_pos_over_time_antigen']['NewPositive'].to_numpy(copy=True)

#idx = pd.date_range("2020-01-27", freq="D",periods = len(DI1)-len(DI2))
#trunc = pd.DataFrame([0 for i in range(len(DI1)-len(DI2))],index=idx)

DI2 = np.concatenate((np.zeros(len(DI1)-len(DI2)),DI2))
DI = DI1 + DI2
#DI = DI1.to_numpy(copy=True)+DI2.to_numpy(copy=True)

N = 5800000
S = []
I = []
R = []
X = []
for i in range(len(DI)):
    if i < 9:
        I.append(sum(DI[0:i]))
    else:
        I.append(sum(DI[i-9:i]))
    if i == 0:
        S.append(N-DI[i])
    else:
        S.append(S[i-1] - DI[i])

    R.append(N-S[i]-I[i])

    X.append([S[i],I[i],R[i]])

idx = pd.date_range("2020-01-27", freq="D",periods = len(DI))
X = pd.DataFrame(X,columns=['S','I','R'],index = idx)
X.to_csv("data/X_basic.csv")
## Vaccinations
#start date
s = pd.to_datetime('2020-12-01')
#start of pandemic
s_p = pd.to_datetime('2020-02-25')
b = 8
num_days = 21


print(gd.vaccine_dict['FaerdigVacc_daekning_DK_prdag']['Kumuleret antal færdigvacc.'])

data = gd.infect_dict['Test_pos_over_time'][s - dt.timedelta(days=b): s + dt.timedelta(days=num_days)]
