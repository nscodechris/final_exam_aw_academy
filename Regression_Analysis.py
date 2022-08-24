
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as oPlotly
import plotly.graph_objs as oGraph



# set options for pandas to see all rows, columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# setting the curr dir path
CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# regression
data = pd.read_csv(CURR_DIR_PATH + "//countries.csv")


def liner_regre(x_data, y_data, test_size):

    # making array of list
    x_list = np.array(x_data)
    y_list = np.array(y_data)


    # making 2d array of list
    x_list = np.reshape(x_list, (-1, 1))
    y_list = np.reshape(y_list, (-1, 1))

    # SZ = 100
    # x_list, y_list = data.make_regression(n_samples=SZ, n_features=1,noise=30)
    x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=test_size)


    # setting variable x & y
    # x, y = x_list, y_list
    # print(f"value of x: {len(x)}")
    # # Split the data into training/testing sets
    # x_train = x[5:-split]
    # x_test = x[-split:]
    #
    # # Split the targets into training/testing sets
    # y_train = y[5:-split]
    # y_test = y[-split:]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    # print(x_train, y_train)
    regr.fit(x_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(x_test)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination(correlation): %.2f" % r2_score(y_test, y_pred))



    # Plot outputs
    plt.scatter(x_test, y_test, color="blue")
    plt.plot(x_test, y_pred, color="red", linewidth=3)

    plt.xticks(())
    plt.yticks(())


    # plt.legend(loc="upper right")
    plt.title(f"{x_data.name}, {y_data.name}")
    plt.xlabel(f"{x_data.name}")
    plt.ylabel(f"{y_data.name}")
    plt.show()
    plt.show()


liner_regre(data["population"], data["coal_consumption"], 0.25)


