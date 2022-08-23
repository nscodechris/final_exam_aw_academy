import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine
import psycopg2
import kaggle_log_in as kglog

# setting the curr dir path
CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# set options for pandas to see all rows, columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_data(data_set):
    # downloading the dataset from kaggle
    # getting user name and kaggle api
    os.environ[kglog.user_name] = kglog.user_name_2
    os.environ[kglog.api_key] = kglog.api_key_2
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(data_set, path=CURR_DIR_PATH, unzip=True)


get_data('pralabhpoudel/world-energy-consumption')


def data_to_pandas(file):

    # data-set to pandas
    data = pd.read_csv(CURR_DIR_PATH + file)
    # print(data.head(5))

    # Filter by needed columns

    data = data[["country", "year", "population", "energy_per_gdp", "electricity_generation",
                 "coal_share_energy", "coal_electricity", "coal_consumption",
                 "gas_share_elec", "gas_electricity", "gas_consumption",
                 "nuclear_share_elec", "nuclear_electricity", "nuclear_consumption",
                "solar_share_elec", "solar_electricity", "solar_consumption",
                 "wind_share_elec", "wind_electricity", "wind_consumption"]]
    return data


def sort_year_filter_nan_export_to_csv(data, year, file_name):
    if year == "yes":
        # filter years
        specific_years = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2019]
        year_filtering = data.loc[data['year'].isin(specific_years)]
        # replace NaN values with 0 (zero)
        final_df = year_filtering.replace(np.nan, 0)
        # final_df to csv file
        final_df.to_csv(CURR_DIR_PATH + file_name, index=False)
    elif year == "no":
        # replace NaN values with 0 (zero)
        final_regression = data.replace(np.nan, 0)
        final_regression.to_csv(CURR_DIR_PATH + file_name, index=False)


sort_year_filter_nan_export_to_csv(data_to_pandas("//World Energy Consumption.csv"), "yes", "//final.csv")


def pandas_to_database(dest, table_name):
    df_final_exam = pd.read_csv(CURR_DIR_PATH + dest)
    print(df_final_exam)
    engine = create_engine('postgresql://postgres:Cvmillan10!?@localhost:5432/final_exam')
    df_final_exam.to_sql(table_name, engine, if_exists='replace', index=False)
    with engine.connect() as connection:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN ID_column serial PRIMARY KEY;")


pandas_to_database("//final.csv", "final")




