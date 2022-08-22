import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sqlalchemy import create_engine
import psycopg2



# setting the curr dir path
CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# getting user name and kaggle api
os.environ['christofferbrodin'] = "<christofferbrodin>"
os.environ['b559bd9c50bbaf4deef3bf6794a0601c'] = "<b559bd9c50bbaf4deef3bf6794a0601c>"
api = KaggleApi()
api.authenticate()

# downloading the dataset from kaggle
api.dataset_download_files('pralabhpoudel/world-energy-consumption', path=CURR_DIR_PATH, unzip=True)

# set options for pandas to see all rows, columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# data-set to pandas
data = pd.read_csv(CURR_DIR_PATH + "//World Energy Consumption.csv")
# print(data.head(5))


# Filter by needed columns

data = data[["country", "year", "population", "energy_per_gdp", "electricity_generation",
             "coal_share_energy", "coal_electricity", "coal_consumption",
             "gas_share_elec", "gas_electricity", "gas_consumption",
             "nuclear_share_elec", "nuclear_electricity", "nuclear_consumption",
            "solar_share_elec", "solar_electricity", "solar_consumption",
             "wind_share_elec", "wind_electricity", "wind_consumption"]]



# filter years
specific_years = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2019]
year_filtering = data.loc[data['year'].isin(specific_years)]


# replace NaN values with 0 (zero)
final_df = year_filtering.replace(np.nan, 0)

# print(final_df)



# final_df to csv file
# final_df.to_csv(CURR_DIR_PATH + "//final.csv", index=False)

def pandas_to_database():
    df_final_exam = pd.read_csv(CURR_DIR_PATH + "//final.csv")
    print(df_final_exam)
    engine = create_engine('postgresql://postgres:Cvmillan10!?@localhost:5432/final_exam')
    df_final_exam.to_sql('final', engine, if_exists='replace', index=False)
    with engine.connect() as connection:
        connection.execute("ALTER TABLE final ADD COLUMN ID_column serial PRIMARY KEY;")

# pandas_to_database()




