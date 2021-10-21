
def DownloadData(
        url = 'https://covid19.who.int/WHO-COVID-19-global-data.csv',
        save_dir = ''
        ):
    
    
    # Import relevant packages
    import requests
    import os

    # Set cwd as either specified path or script dir
    if save_dir:
        os.chdir(save_dir)
    else:
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)

    # Download and save file
    r = requests.get(url, allow_redirects=True)
    url_content = r.content
    open('Covid19_data_daily_by_country.csv', 'wb').write(url_content)

    return


def ReadDataFromCsvFile(
        file_path,
        ):
    
    # Import relevant packages
    import pandas as pd
    
    # Read csv file into pandas dataframe
    data = pd.read_csv(file_path)

    # >>> Data preperation <<<
    attribute_names = data.columns
    attribute_dict = dict(zip(attribute_names, range(len(attribute_names))))
    
    # --- Special cases ---
    # The countrycode of Namibia is missing, and is set to NA.
    data['Country_code'].fillna('NA', inplace=True)
    
    # The observations without a country accounts for a total of only 764 new positives, hence it is removed
    data = data[data['Country'] != 'Other']
    
    # Country and country codes is essentially the same, thus they are paired and their columns replaced by a single column with integer keys (If one wanted, it could be considered to use actual country codes for more flexibility).
    countries = set(zip(data['Country_code'], data['Country']))
    country_dict = dict(zip([val for sublist in countries for val in sublist], [i for i in range(len(countries)) for _ in range(2)]))
    data = data.drop('Country_code', axis = 1)
    
    # Enumerate the WHO regions
    WHOregions = list(set(data["WHO_region"]))
    WHOregion_dict = dict(zip(WHOregions, range(len(WHOregions))))
    
    # Enumerate dates
    dates = sorted(set(data[attribute_names[0]]))
    date_dict = dict(zip(dates, range(len(dates))))

    # Use the constructed dicts to change all string values to integer values - This makes the dataset easily usable with machine learning methods.
    data.replace({'Country': country_dict, 'Date_reported': date_dict, 'WHO_region': WHOregion_dict}, inplace=True)

    # Return the processed dataframe along with the necessary keys.
    return data, attribute_dict, date_dict, country_dict, WHOregion_dict


def ExtractContries(
        dataframe,
        countries,
        country_dict
        ):
    
    # Check if only 1 country is specified.
    if type(countries) == str:
        return dataframe.loc[dataframe['Country'] == country_dict[countries]].reset_index(drop=True)
        
    elif type(countries) == list:
        # Prepare return dict
        extracted_countries = dict()
        for country in countries:
            extracted_countries[country] = dataframe.loc[dataframe['Country'] == country_dict[country]].reset_index(drop=True)
            
        return extracted_countries
    
    else:
        raise(ValueError)
    

def SIRdataframe(
        dataframe,
        pop = True,
        gamma = 7,
        dark_number_scalar = 1,
        standardize = False
        ):
    
    # Import relevant packages
    import pandas as pd
    
    if pop:
        N = 5600000 # Should be changed to actual populationvalues
    elif type(pop) == int:
        N = pop
    else:
        raise(ValueError)
        
    SIR_data = pd.DataFrame()
    # SIR_data = pd.DataFrame((dataframe['Date_reported']))
    
    # Compute compartments
    SIR_data['S'] = N - (dataframe['New_cases'].cumsum() * dark_number_scalar)
    SIR_data['I'] = dataframe['New_cases'].rolling(min_periods=1, window=gamma).sum().astype('int64') * dark_number_scalar
    SIR_data['R'] = N - (SIR_data['S'] + SIR_data['I'])
     
    if standardize:
        SIR_data = (SIR_data - SIR_data.mean())/SIR_data.std()
    
    return SIR_data
    
