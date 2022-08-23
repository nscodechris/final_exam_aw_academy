import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


# setting the curr dir path
CURR_DIR_PATH = os.path.dirname(os.path.realpath(__file__))


data = pd.read_csv(CURR_DIR_PATH + "//final.csv")
# print(data.info)

world_data = data.loc[data["country"] == "World"]

x_data =world_data["year"]
y_data = world_data["coal_consumption"]

print(x_data)
print(y_data)

# making array of list
x_list = np.array(x_data)
y_list = np.array(y_data)

# making 2d array of list
x_list = np.reshape(x_list, (-1, 1))
y_list = np.reshape(y_list, (-1, 1))

# setting variable x & y
x, y = x_list, y_list

# Split the data into training/testing sets

x_train = x[5:-30]
x_test = x[-30:]

# Split the targets into training/testing sets
y_train = y[5:-30]
y_test = y[-30:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print("Coefficients: \n", regr.coef_)



# Plot outputs
plt.figure(figsize=(15,10))
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