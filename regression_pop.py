import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('World Energy Consumption.csv')

df_primary = df[['gdp', 'biofuel_consumption', 'population']]
filter_nan = df_primary.replace(np.NaN,0)
#print(filter_nan)

filter_nan.to_csv('final.csv',index=False)
df_primary = pd.read_csv('final.csv')

gdp_filter = df['gdp'].replace(np.NaN,0)
biofuel_filter = df['biofuel_consumption'].replace(np.NaN,0)
poupulation_filter = df['population'].replace(np.NaN,0)
#print(poupulation_filter)

X = np.array(df_primary['gdp']).reshape(-1, 1)
y = np.array(df_primary['population']).reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

regr = linear_model.LinearRegression()

regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

print("Coefficients: \n", regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))

plt.scatter(x_test, y_test, color="red")
plt.plot(x_test, y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


