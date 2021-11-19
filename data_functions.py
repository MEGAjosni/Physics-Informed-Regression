
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
    
    import os
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    import pandas as pd
    
    data = pd.read_csv('Covid19_data_daily_by_country.csv')
    
    country_codes = pd.read_csv('country_codes.csv', index_col=0)
    country_codes.fillna('NA', inplace=True)
    codes = dict(zip(country_codes['Alpha2Code'].to_numpy(), country_codes.index))
    
    # Enumerate the WHO regions
    WHOregions = list(set(data["WHO_region"]))
    WHOregion_dict = dict(zip(WHOregions, range(len(WHOregions))))
    
    # Enumerate dates
    dates = sorted(set(data['Date_reported']))
    date_dict = dict(zip(dates, range(len(dates))))
    
    # Fix Namibia being assigned nan due to country code
    data['Country_code'].fillna('NA', inplace=True)
    data = data[data['Country'] != 'Other']
    
    # Country and country code are essentially the same thing, thus country name is dropped
    data = data.drop('Country', axis=1)
    
    # Countries without a well defined country code are removed
    data.replace({'Country_code': codes, 'Date_reported': date_dict, 'WHO_region': WHOregion_dict}, inplace=True)
    
    data = data[data['Country_code'].apply(lambda x: not isinstance(x, str))]
    
    # Return the processed dataframe along with the necessary keys.
    return data, date_dict, WHOregion_dict


def GetCountryCode(
        key
        ):
    
    import pandas as pd
    
    country_codes = pd.read_csv('country_codes.csv', index_col=0)
    
    numeric_code = []
    
    for code_type in country_codes.columns:
        numeric_code += country_codes.index[country_codes[code_type] == key].to_list()
    
    if len(numeric_code) == 0:
        print('Warning! No country found using the key:', key)
        return None
    elif len(numeric_code) == 1:
        return numeric_code[0]
    else:
        raise(ValueError)


def ExtractContries(
        dataframe,
        countries
        ):
    
    # Check if only 1 country is specified.
    if type(countries) == int:
        return dataframe.loc[dataframe['Country_code'] == countries].reset_index(drop=True)
        
    elif type(countries) == list:
        # Prepare return dict
        extracted_countries = dict()
        for country in countries:
            extracted_countries[country] = dataframe.loc[dataframe['Country_code'] == country].reset_index(drop=True)
            
        return extracted_countries
    
    else:
        raise(ValueError)
    
    
def SIRdataframe(
        dataframe,
        N,
        gamma = 1/9,
        dark_number_scalar = 1,
        standardize = False
        ):
    
    # Import relevant packages
    import pandas as pd
        
    SIR_data = pd.DataFrame()
    # SIR_data = pd.DataFrame((dataframe['Date_reported']))
    
    # Compute compartments
    SIR_data['S'] = N - (dataframe['New_cases'].cumsum() * dark_number_scalar)
    SIR_data['I'] = dataframe['New_cases'].rolling(min_periods=1, window=int(1/gamma)).sum().astype('int64') * dark_number_scalar
    SIR_data['R'] = N - (SIR_data['S'] + SIR_data['I'])
     
    if standardize:
        SIR_data = (SIR_data - SIR_data.mean())/SIR_data.std()
    
    return SIR_data
    

def getPopulation(
        countries,
        ):
    
    import pandas as pd
    
<<<<<<< Updated upstream
    print('>>>>> Warning: The validity of the getPopulation function is questionable! <<<<<')
    
    pop = pd.read_csv('worldpop.csv')
    pop_dict = pd.Series(pop['population'].values,index=pop['country']).to_dict()
    
    if type(countries) == int:
=======
    pop = pd.read_csv('worldpop.csv')
    pop_dict = pd.Series(pop['population'].values,index=pop['country']).to_dict()
    
    if type(countries) == str:
>>>>>>> Stashed changes
        return pop_dict[countries]
    else:    
        d = dict()
        
        for country in countries:
            d[country] = pop_dict[country]
    
        return d
