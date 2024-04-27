from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore")

# load data
df = pd.read_csv('dataset.csv')
print("\n Data loaded successfully! \n")
print(df.head(10), df.shape, df.dtypes)

# plot the data to see the relationship
df.plot(x='Investment', y='Return', style='o')
plt.title('Investment vs. Return')
plt.xlabel('Investment')
plt.ylabel('Return')
plt.savefig('chart1.png')
plt.show()

# split into features and target
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# split into train and test as 30% test data and 70% training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# reshape to vector format and turn no float
x_train = x_train.reshape(-1, 1).astype(np.float32)

# create model and train model
model = LinearRegression()
model.fit(x_train, y_train)
print("\n Training complete! \n")

# print coefficients
print('b0 (intercept_):', model.intercept_)
print('b1 (coefficient_):', model.coef_)

# y = b1*x + b0
regression_line = model.coef_ * x + model.intercept_

# plot the regression line
plt.scatter(x, y)
plt.title('Investment vs. Return')
plt.xlabel('Investment')
plt.ylabel('Return')
plt.plot(x, regression_line, color='red')
plt.savefig('chart2.png')
plt.show()

# predict the test data
y_pred = model.predict(x_test)
df_values = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df_values)

# plot bar chart of predicted vs actual side by side
plt.figure(figsize=(10, 6))
plt.bar(np.arange(len(y_test)) - 0.25, y_test, color='blue', width=0.2, label='Actual')
plt.bar(np.arange(len(y_test)) + 0.25, y_pred, color='red', width=0.2, label='Predicted')
plt.xlabel('Data Points')
plt.ylabel('Value')
plt.title('Predicted vs Actual')
plt.xticks(np.arange(len(y_test)) + 0.25 - 0.125, range(len(y_test)))
plt.legend()
plt.savefig('chart3.png')
plt.show()

# print evaluation metrics
print("\n")
print("MAE (mean absolute error):", mean_absolute_error(y_test, y_pred))
print("MSE (mean squared error):", mean_squared_error(y_test, y_pred))
print("RMSE (root mean squared error):", math.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 score:", r2_score(y_test, y_pred))

# predict investment from input
input_inv = input("\nEnter investment amount: ")
input_inv = float(input_inv)
inv = np.array([[input_inv]])
inv = inv.reshape(-1, 1).astype(np.float32)
pred_score = model.predict(inv)

# print predicted return
print("\nInvestment: ", input_inv)
print("Return: ", pred_score[0].round(2))