import ETL_data
import kaggle_log_in as kglog
import Regression_dash as rd

def main():

    # ETL_data.get_data('pralabhpoudel/world-energy-consumption', kglog.user_name, kglog.user_name_2,
    #                   kglog.api_key, kglog.api_key_2)

    # select no if extracting to power bi, for all data, yes for filter data
    ETL_data.data_to_pandas("//World Energy Consumption.csv", "yes")

    # select etl.power_bi for power bi, etl.final_df for filter_data_frame
    ETL_data.cleaning_data_to_csv(ETL_data.etl.final_df, '//countries.csv', '//continents.csv'
                                  , '//non_countries.csv')  # enter 3 file names like: "//countries.csv"

    ETL_data.pandas_to_database("//countries.csv", "final", kglog.postgress_pass)


if __name__ == "__main__":
    main()
