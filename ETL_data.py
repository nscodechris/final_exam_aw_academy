import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine
import psycopg2
import pycountry
import seaborn as sns


class EtlData:
    def __init__(self):
        self.data = pd.DataFrame()
        self.final_df = pd.DataFrame()


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


def data_to_pandas(file, filter_year):
    # data-set to pandas
    etl.data = pd.read_csv(CURR_DIR_PATH + file)
    print(etl.data.head(100))
    # Filter by needed columns
    etl.data = etl.data[["country", "year", "population", "energy_per_gdp", "gdp", "electricity_generation",
                 "coal_share_energy", "coal_electricity", "coal_consumption",
                 "gas_share_elec", "gas_electricity", "gas_consumption",
                 "nuclear_share_elec", "nuclear_electricity", "nuclear_consumption",
                 "solar_share_elec", "solar_electricity", "solar_consumption",
                 "wind_share_elec", "wind_electricity", "wind_consumption"]]

    # # Filter by selected countries
    # countries = [
    #     "Brazil", "China", "Denmark", "India",
    #     "Italy", "Japan", "North Korea", "Russia", "Saudi Arabia", "United States", "Sweden"]
    #
    # etl.data = etl.data.loc[etl.data['country'].isin(countries)]

    if filter_year == "yes":
        # filter years
        specific_years = [i for i in range(1960, 2020)]  # [1985, 1990, 1995, 2000, 2005, 2010, 2019]
        # print(specific_years)
        year_filtering = etl.data.loc[etl.data['year'].isin(specific_years)]  # .isin(specific_years)
        # replace NaN values with 0 (zero)
        etl.final_df = year_filtering.replace(np.nan, 0)
        return etl.final_df
        # final_df to csv file
    elif filter_year == "no":
        # replace NaN values with 0 (zero)
        etl.final_df = etl.data.replace(np.nan, 0)
        return etl.final_df


def cleaning_data_to_csv(data, country_csv, continent_csv, non_countries_csv):
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
    print(data_all_countries.head(100))

    data_all_countries.to_csv(CURR_DIR_PATH + country_csv, index=False)
    data_all_continents.to_csv(CURR_DIR_PATH + continent_csv, index=False)
    data_non_countries_list.to_csv(CURR_DIR_PATH + non_countries_csv, index=False)


def pandas_to_database(file_name, table_name, postgress_pass):
    df_final_exam = pd.read_csv(CURR_DIR_PATH + file_name)
    # print(df_final_exam)
    engine = create_engine(f'postgresql://postgres:{postgress_pass}@localhost:5432/final_exam')
    df_final_exam.to_sql(table_name, engine, if_exists='replace', index=False)
    with engine.connect() as connection:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN ID_column serial PRIMARY KEY;")

