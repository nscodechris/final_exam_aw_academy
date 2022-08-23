
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import pycountry



# set options for pandas to see all rows, columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# setting the curr dir path
CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


data = pd.read_csv(CURR_DIR_PATH + "//final.csv")

# Getting unique country values, to see countries in data_set
unique_country_values = data['country'].unique()
# print(unique_country_values)

countries_in_world = []
countries_to_use = []
non_countries = []

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

data = data.apply(lambda x: x[data["country"].isin(countries_to_use)])
# data["population"] = data["population"].astype(int)
print(non_countries)
print(countries_to_use)
# print(data)

def liner_regre(x_data, y_data, split):

    # making array of list
    x_list = np.array(x_data)
    y_list = np.array(y_data)


    # making 2d array of list
    x_list = np.reshape(x_list, (-1, 1))
    y_list = np.reshape(y_list, (-1, 1))

    # setting variable x & y
    x, y = x_list, y_list
    print(f"value of x: {len(x)}")
    # Split the data into training/testing sets
    x_train = x[:-split]
    x_test = x[-split:]

    # Split the targets into training/testing sets
    y_train = y[:-split]
    y_test = y[-split:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    # print(x_train, y_train)
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)



    # Plot outputs
    plt.scatter(x_test, y_test, color="blue")
    plt.plot(x_test, y_pred, color="red", linewidth=3)

    plt.xticks(())
    plt.yticks(())


    # plt.legend(loc="upper right")
    plt.title(f"{x_data.name}, {y_data.name}")
    plt.xlabel(f"{x_data.name}")
    plt.ylabel(f"{y_data.name}")
    # plt.show()
    plt.show()

# liner_regre(data["population"], data["coal_consumption"], 800)

