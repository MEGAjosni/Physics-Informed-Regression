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

dataDK = ExtractContries(data, 'Denmark', country_dict)

SIRdata = SIRdataframe(dataDK, dark_number_scalar = 10)

fig = px.line(SIRdata)
fig.show()


dataNord = ExtractContries(data, ['Denmark', 'Sweden', 'Norway'], country_dict)