import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# For graphing
import plotly.express as px # For plotting and visualization
import plotly.io as pio
pio.renderers.default = "browser" # This set the default render as browser (This is not necessary if not using Spyder) 

import numpy as np
from data_functions import ReadDataFromCsvFile, ExtractContries, SIRdataframe, getPopulation, GetCountryCode, DownloadData
from sim_functions import LeastSquareModel, SimulateModel


data, date_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")
country = GetCountryCode('Denmark')
dataDK = ExtractContries(data, country)
SIRdata = SIRdataframe(dataDK, dark_number_scalar = 1, standardize=False)


t = np.arange(10)

A = np.zeros((10, 10))

for i in range(1, 11):
    for j in range(1, 11):
        mp_used = np.array([i/10, j/10])
        SIR = SimulateModel(t, [5600000, 100000, 0], mp_used)
        mp = LeastSquareModel(t, SIR)
        A[i-1, j-1] = np.mean(1 - mp/mp_used)

    print(i)

x = np.linspace(0.1, 1, 10)
y = np.linspace(0.1, 1, 10)

X, Y = np.meshgrid(x, y)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, A)
ax.set_title('Surface plot')
plt.show()



traindays = 14

t = np.arange(traindays)
SIR = SIRdata.iloc[t]

while t[-1] <= date_dict[list(date_dict)[-1]]:
    mp = list(LeastSquareModel(t, SIRdata.iloc[t], fix_params=[0, 1/9]))
    mp += [1/9]
    t += traindays
    SIR = np.concatenate((SIR, SimulateModel(t, list(SIRdata.iloc[t[0]]), mp)), axis=0)
    
print(SIR)

fig = px.line(y=SIR[:, 1])
fig.add_scatter(y=SIRdata['I'], mode='lines')
fig.show()
