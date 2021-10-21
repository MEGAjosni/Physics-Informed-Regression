import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# For graphing
import plotly.express as px # For plotting and visualization
import plotly.io as pio
pio.renderers.default = "browser" # This set the default render as browser (This is not necessary if not using Spyder) 

from data_functions import ReadDataFromCsvFile, ExtractContries, SIRdataframe

data, attribute_dict, date_dict, country_dict, WHOregion_dict = ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")

dataDK = ExtractContries(data, 'Sweden', country_dict)

SIRdata = SIRdataframe(dataDK, dark_number_scalar = 1, standardize=True)

fig = px.line(SIRdata)
fig.show()

nordic = ['Denmark', 'Sweden', 'Norway']

nordDict = ExtractContries(data, nordic, country_dict)

nordData = []
for df in nordDict.values():
    nordData += [SIRdataframe(df, standardize=True)]

nordData = sum(nordData)

fig = px.line(nordData)
fig.show()