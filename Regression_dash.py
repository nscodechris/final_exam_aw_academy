
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(CURR_DIR_PATH + "//countries.csv")


# set options for pandas to see all rows, columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)




# setting the curr dir path

# regression

data_1960_1970 = data.loc[data['year'].isin(i for i in range(1960, 1970))]
data_1970_1980 = data.loc[data['year'].isin(i for i in range(1970, 1980))]
data_1980_1990 = data.loc[data['year'].isin(i for i in range(1980, 1990))]
data_1990_2000 = data.loc[data['year'].isin(i for i in range(1990, 2000))]
data_2000_2010 = data.loc[data['year'].isin(i for i in range(2000, 2010))]


app = Dash(__name__)

year = {'1960-1969': data_1960_1970,
          '1970-1979': data_1970_1980,
          '1980-1989': data_1980_1990,
          '1990-1999': data_1990_2000,
        '2000-2010': data_2000_2010
          }


energy = {'coal_consumption': data.coal_consumption,
          'gas_consumption': data.gas_consumption,
          'nuclear_consumption': data.nuclear_consumption,
          'solar_consumption': data.solar_consumption,
          'wind_consumption': data.wind_consumption,
          'coal_electricity': data.coal_electricity,
          'gas_electricity': data.gas_electricity,
          'nuclear_electricity': data.nuclear_electricity,
          'solar_electricity': data.solar_electricity,
          'wind_electricity': data.wind_electricity

          }

x_axis = {'population': data.population,
          'energy_per_gdp': data.energy_per_gdp,
          'electricity_generation': data.electricity_generation,
          'gdp': data.gdp
          }


app.layout = html.Div([
    # html.H4("Predicting restaurant's revenue"),
    # html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=["coal_consumption", "gas_consumption", "nuclear_consumption", "solar_consumption", "wind_consumption",
                 "coal_electricity", "gas_electricity", "nuclear_electricity", "solar_electricity", "wind_electricity"],
        value='coal_consumption',
        clearable=False
    ),
    dcc.Graph(id="graph"),
    dcc.Dropdown(
        id='dropdown_2',
        options=["population", "energy_per_gdp", "electricity_generation", 'gdp'],
        value='gdp',
        clearable=False
    ),
    dcc.Dropdown(
        id='dropdown_3',
        options=["1960-1969", "1970-1979", "1980-1989", '1990-1999', '2000-2010'],
        value='1960-1969',
        clearable=False
    ),




])


@app.callback(
    Output("graph", "figure"),
    Input('dropdown', "value"),
    Input('dropdown_2', "value"),
    Input('dropdown_3', "value"))
def train_and_display(energy_choice, x_axis_choice, year_choice):
    # detect outliers and remove

    mean = year[year_choice][x_axis_choice].mean()
    sd = year[year_choice][x_axis_choice].std()
    # mean_2 = year[year_choice][energy_choice].mean()
    # sd_2 = year[year_choice][energy_choice].std()
    #
    year[year_choice] = year[year_choice][(year[year_choice][x_axis_choice] <= mean + (3 * sd))]
    # year[year_choice] = year[year_choice][(year[year_choice][energy_choice] <= mean_2 + (3 * sd_2))]

    values_use = []
    group = year[year_choice].groupby('country')[[x_axis_choice]].mean()
    z = np.abs(stats.zscore(group[x_axis_choice]))
    # print(z)
    for i in group[x_axis_choice]:
        values_use.append(i)
    list_pandas = pd.Series(i for i in values_use)

    values_use_2 = []
    group_energy = year[year_choice].groupby('country')[[energy_choice]].mean()
    for i in group_energy[energy_choice]:
        values_use_2.append(i)
    list_pandas_2 = pd.Series(i for i in values_use_2)

    X = list_pandas.values[:, None]
    # print(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X,  list_pandas_2, test_size=0.20, shuffle=False)  #  , shuffle=False)

    # X = year[year_choice][x_axis_choice].values[:, None]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, year[year_choice][energy_choice].values, test_size=0.25)

    model = LinearRegression()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    y_pred = model.predict(X_test)

    # The coefficients
    print("Coefficients: \n", model.coef_)
    # The mean squared error
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print("Coefficient of determination(correlation): %.2f" % r2_score(y_test, y_pred))
    coef_detr_corr = round(r2_score(y_test, y_pred), 2)
    mean_sq_error = round(mean_squared_error(y_test, y_pred), 2)

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name='prediction')
    ])
    fig.update_layout(title_text=f"<b>Coefficient of determination(correlation): {coef_detr_corr}")

    return fig


app.run_server(debug=True)


