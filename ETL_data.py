import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine
import psycopg2
import pycountry


class EtlData:
    def __init__(self):
        self.data = pd.DataFrame()
        self.final_df = pd.DataFrame()
        self.regression_df = pd.DataFrame()


etl = EtlData()

# glob variables get countries & continents
countries_in_world = []
countries_to_use = []
non_countries = []
continents = ['Africa', 'Asia Pacific', 'Europe', 'North America', 'South & Central America', 'Other South America',
              'Other Asia & Pacific']

# setting the curr dir path
CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# set options for pandas to see all rows, columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_data(data_set, user_name, user_name_2, api_key, api_key_2):
    # downloading the dataset from kaggle
    # getting user name and kaggle api
    os.environ[user_name] = user_name_2
    os.environ[api_key] = api_key_2
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(data_set, path=CURR_DIR_PATH, unzip=True)


# get_data('pralabhpoudel/world-energy-consumption', kglog.user_name, kglog.user_name_2, kglog.api_key, kglog.api_key_2)


def data_to_pandas(file, filter_year):
    # data-set to pandas
    etl.data = pd.read_csv(CURR_DIR_PATH + file)
    # Filter by needed columns
    etl.data = etl.data[["country", "year", "population", "energy_per_gdp", "electricity_generation",
                 "coal_share_energy", "coal_electricity", "coal_consumption",
                 "gas_share_elec", "gas_electricity", "gas_consumption",
                 "nuclear_share_elec", "nuclear_electricity", "nuclear_consumption",
                 "solar_share_elec", "solar_electricity", "solar_consumption",
                 "wind_share_elec", "wind_electricity", "wind_consumption"]]

    if filter_year == "yes":
        # filter years
        specific_years = 1960  # [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2019]
        year_filtering = etl.data.loc[etl.data['year'] >= specific_years]  # .isin(specific_years)]
        # replace NaN values with 0 (zero)
        etl.final_df = year_filtering.replace(np.nan, 0)
        return etl.final_df
        # final_df to csv file
    elif filter_year == "no":
        # replace NaN values with 0 (zero)
        etl.regression_df = etl.data.replace(np.nan, 0)
        return etl.regression_df


# data_to_pandas("//World Energy Consumption.csv", "yes")

def cleaning_data_to_csv(data):
    # replace NaN values with 0 (zero)
    data = data.replace(np.nan, 0)
    # Getting unique country values, to see countries in data_set
    unique_country_values = data['country'].unique()
    # get country name from pycountry to countries_in_world list
    for country in pycountry.countries:
        # print(country.name)
        countries_in_world.append(country.name)
    # checking if countries_in_world list is in data_set
    for i in countries_in_world:
        if i in unique_country_values:
            countries_to_use.append(i)
    # non countries from data_set - getting out the continents
    for q in unique_country_values:
        if q not in countries_in_world:
            non_countries.append(q)

    data_all_countries = data.apply(lambda x: x[data["country"].isin(countries_to_use)])
    data_all_continents = data.apply(lambda x: x[data["country"].isin(continents)])
    data_non_countries_list = data.apply(lambda x: x[data["country"].isin(non_countries)])

    data_all_countries.to_csv(CURR_DIR_PATH + '//countries.csv', index=False)
    data_all_continents.to_csv(CURR_DIR_PATH + '//continents.csv', index=False)
    data_non_countries_list.to_csv(CURR_DIR_PATH + '//non_countries.csv', index=False)


# cleaning_data_to_csv(etl.final_df)



def pandas_to_database(file_name, table_name, postgress_pass):
    df_final_exam = pd.read_csv(CURR_DIR_PATH + file_name)
    # print(df_final_exam)
    engine = create_engine(f'postgresql://postgres:{postgress_pass}@localhost:5432/final_exam')
    df_final_exam.to_sql(table_name, engine, if_exists='replace', index=False)
    with engine.connect() as connection:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN ID_column serial PRIMARY KEY;")


# pandas_to_database("//final.csv", "final", kglog.postgress_pass)





