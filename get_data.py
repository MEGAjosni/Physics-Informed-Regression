import requests
import zipfile
import pandas as pd
import os
import shutil
import pickle

# ***** Description *****
#
# Get newest Covid-19 data from ssi.dk.
#
# Importing this scripts allows access to the dictionary, data_dict, which contains all the
# data as 'pandas' dataframes. The dataframes are named after the file they are read from.
#
# Specific parts of the date-indexed data can be accessed by;
#
#       data_dict[data_name][start_day : end_day]
#
# --- Example ---
# Say you want the data from 'Deaths_over_time', and want the data from 2021-01-01 to 2021-01-31
# (both inclusive). You get this by the command;
#
#       data_dict['Deaths_over_time']['2021-01-01' : '2021-01-31']
#
# ***** End *****


# Folder where data should be stored
data_dir = os.getcwd() + '\\data\\infection\\'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Get zipped data folder
url = 'https://files.ssi.dk/covid19/overvagning/data/overvaagningsdata-covid19-08062021-f67f'  # Zip download-link
# Url changes daily check https://covid19.ssi.dk/overvagningsdata/download-fil-med-overvaagningdata to find newest link.
zip_name = url[-39:] + '.zip'
r = requests.get(url, allow_redirects=True)
open(data_dir + zip_name, 'wb').write(r.content)

# Unzip and extract files
with zipfile.ZipFile(data_dir + zip_name, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

os.remove(data_dir + zip_name)

infect_dict: dict = {}

for f in os.listdir(data_dir):
    if f.endswith('.csv'):
        # Load data into dataframe
        x = pd.read_csv(
            filepath_or_buffer=data_dir + f,
            sep=';',
            thousands='.',
            decimal=',',
            engine='python'
        )

        # If data is date-indexed, the date is the first column
        date_name = x.columns[0]

        # If data is dateindexed convert to datetime64[ns]
        if len(str(x[date_name][0])) == 10:

            # Check different formats of time, yyyy-mm-dd and dd-mm-yyyy
            format1 = sum([str(x[date_name][0])[i] == 'yyyy-mm-dd'[i] for i in range(10)]) == 2
            format2 = sum([str(x[date_name][0])[i] == 'dd-mm-yyyy'[i] for i in range(10)]) == 2

            # If either of the formats are present convert to datetime64[ns]
            if format1 or format2:
                j = len(x[date_name]) - 1

                # Remove totals from bottom
                while True:
                    if len(str(x[date_name][j])) == 10:

                        format1 = sum([str(x[date_name][j])[i] == 'yyyy-mm-dd'[i] for i in range(10)]) == 2
                        format2 = sum([str(x[date_name][j])[i] == 'dd-mm-yyyy'[i] for i in range(10)]) == 2
                        format3 = sum([str(x[date_name][j])[i] == 'dd/mm/yyyy'[i] for i in range(10)]) == 2

                        if format1 or format2 or format3:
                            break
                    else:
                        x = x.drop(index=[j])
                    j += -1

                x[date_name] = pd.to_datetime(x[date_name], dayfirst=True)
                x = x.set_index(pd.DatetimeIndex(x[date_name]))

                # As list is now indexed with dates, get rid of the datecolumn
                x = x.drop(columns=[date_name])

        # Insert data to dictionary
        infect_dict[f[0:-4]] = x

a_file = open('infect_data.pkl', 'wb')
pickle.dump(infect_dict, a_file)
a_file.close()

# >>> Get vaccine data <<<
# Folder where data should be stored
data_dir = os.getcwd() + '\\data\\vaccination\\'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Get zipped data folder
url = 'https://files.ssi.dk/covid19/vaccinationsdata/zipfil/vaccinationsdata-dashboard-covid19-16062021-re43'  # Zip download-link
r = requests.get(url, allow_redirects=True)
open(data_dir + 'covid19-vaccinationsdata-29032021-lb1f.zip', 'wb').write(r.content)

# Unzip and extract files
with zipfile.ZipFile(data_dir + 'covid19-vaccinationsdata-29032021-lb1f.zip', 'r') as zip_ref:
    zip_ref.extractall(data_dir)

os.remove(data_dir + 'covid19-vaccinationsdata-29032021-lb1f.zip')

# Since these files are inside a folder they must be moved to the correct folder
for filename in os.listdir(data_dir + 'Vaccine_DB\\'):
    shutil.move(os.path.join(data_dir + 'Vaccine_DB\\', filename), os.path.join(data_dir, filename))

vaccine_dict: dict = {}

for f in os.listdir(data_dir):
    if f.endswith('.csv'):
        # Load data into dataframe
        x = pd.read_csv(
            filepath_or_buffer=data_dir + f,
            sep=',',
            thousands='.',
            decimal=',',
            engine='python'
        )

        # If data is date-indexed, the date is the first column
        date_name = x.columns[0]

        # If data is dateindexed convert to datetime64[ns]
        if len(str(x[date_name][0])) == 10:

            # Check different formats of time, yyyy-mm-dd and dd-mm-yyyy
            format1 = sum([str(x[date_name][0])[i] == 'yyyy-mm-dd'[i] for i in range(10)]) == 2
            format2 = sum([str(x[date_name][0])[i] == 'dd-mm-yyyy'[i] for i in range(10)]) == 2

            # If either of the formats are present convert to datetime64[ns]
            if format1 or format2:
                j = len(x[date_name]) - 1

                # Remove totals from bottom
                while True:
                    if len(str(x[date_name][j])) == 10:

                        format1 = sum([str(x[date_name][j])[i] == 'yyyy-mm-dd'[i] for i in range(10)]) == 2
                        format2 = sum([str(x[date_name][j])[i] == 'dd-mm-yyyy'[i] for i in range(10)]) == 2

                        if format1 or format2:
                            break
                    else:
                        x = x.drop(index=[j])
                    j += -1

                x[date_name] = pd.to_datetime(x[date_name], dayfirst=True)
                x = x.set_index(pd.DatetimeIndex(x[date_name]))

                # As list is now indexed with dates, get rid of the datecolumn
                x = x.drop(columns=[date_name])

        # Insert data to dictionary
        vaccine_dict[f[0:-4]] = x

shutil.rmtree(data_dir + 'Vaccine_DB')
