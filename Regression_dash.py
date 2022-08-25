
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


CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(CURR_DIR_PATH + "//countries.csv")

# set options for pandas to see all rows, columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# setting the curr dir path

# regression



app = Dash(__name__)

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
        value='population',
        clearable=False
    ),




])


@app.callback(
    Output("graph", "figure"),
    Input('dropdown', "value"),
    Input('dropdown_2', "value"))
def train_and_display(name, sort):

    X = x_axis[sort].values[:, None] #  df.population.values[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        X, energy[name], test_size=0.30)

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
    fig.update_layout(title_text=f"<b>Coefficient of determination(correlation): {coef_detr_corr}\n"
                                 f"<b>                      Mean squared error: {mean_sq_error}")

    return fig


app.run_server(debug=True)


