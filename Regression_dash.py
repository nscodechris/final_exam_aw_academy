
from dash import Dash, dcc, html, Input, Output
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import pandas as pd

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
          'nuclear_consumption': data.nuclear_consumption}

x_axis = {'population': data.population,
          'energy_per_gdp': data.energy_per_gdp,
          'electricity_generation': data.electricity_generation}


app.layout = html.Div([
    # html.H4("Predicting restaurant's revenue"),
    # html.P("Select model:"),
    dcc.Dropdown(
        id='dropdown',
        options=["coal_consumption", "gas_consumption", "nuclear_consumption"],
        value='coal_consumption',
        clearable=False
    ),
    dcc.Graph(id="graph"),
    dcc.Dropdown(
        id='dropdown_2',
        options=["population", "energy_per_gdp", "electricity_generation"],
        value='population',
        clearable=False
    ),

])


@app.callback(
    Output("graph", "figure"),
    Input('dropdown', "value"),
    Input('dropdown_2', "value"),)
def train_and_display(name, sort):

    # df = data # replace with your own data source
    X = x_axis[sort].values[:, None] #  df.population.values[:, None]
    X_train, X_test, y_train, y_test = train_test_split(
        X, energy[name], random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = go.Figure([
        go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
        go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
        go.Scatter(x=x_range, y=y_range, name='prediction')
    ])

    return fig

app.run_server(debug=True)