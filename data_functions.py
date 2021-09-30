
def DownloadData(
        url = 'https://covid19.who.int/WHO-COVID-19-global-data.csv',
        save_dir = ''
        ):
    

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
    data = data.drop(["Country_code"], axis=1)

    # >>> Data preperation <<<
    attribute_names = data.columns
    attribute_dict = dict(zip(attribute_names, range(len(attribute_names))))
    
    country_names = [name for name in list(set(data["Country"])) if (type(name) == str and not str.isspace(name))]
    country_dict = dict(zip(country_names, range(len(country_names))))
    
    WHOregions = [name for name in list(set(data["WHO_region"])) if (type(name) == str and not str.isspace(name))]
    WHOregion_dict = dict(zip(WHOregions, range(len(WHOregions))))
    
    dates = sorted(set(data[attribute_names[0]]))
    date_dict = dict(zip(dates, range(len(dates))))

    data.replace({"Country": country_dict, "Date_reported": date_dict}, inplace=True)
    data = data.to_numpy()

    return data, attribute_dict, date_dict, country_dict, WHOregion_dict

ReadDataFromCsvFile("Covid19_data_daily_by_country.csv")