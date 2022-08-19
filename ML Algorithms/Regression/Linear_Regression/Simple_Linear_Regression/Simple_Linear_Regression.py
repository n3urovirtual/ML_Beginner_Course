# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Create a plot
plt1 = plt.figure(1)
plt.scatter(X, Y)
plt.xlabel("Years of Experience")
plt.ylabel("Annual Salary in $")
plt1.show()

# Split the dataset into the Training and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1 / 4)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fit Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)  # training the algorithm

# Predict the Test set results
Y_predicted = regressor.predict(X_test)

# Compare the actual output values for X_test with the predicted values
comparison = pd.DataFrame(
    {"Actual Salary": Y_test.flatten(), "Predicted Salary": Y_predicted.flatten()}
)

# Evaluate the performance of the algorithm
from sklearn.metrics import mean_squared_error, r2_score

slope = regressor.coef_
intercept = regressor.intercept_
mse = mean_squared_error(Y_test, Y_predicted)
variance = np.var(Y_predicted)
r2 = r2_score(Y_test, Y_predicted)

# Real Life example
"""Linear Regression Equation is Y= mX+b
assume that the years of experience are 7.5 (X=7.5)
Y= dependent variable, annual salary, m=slope, X= independent variable, years of exp. , b=intercept"""
Estimated_salary = slope * 7.5 + intercept
Y_est = regressor.predict([[7.5]])

# Make a plot to visualize the training test results
plt2 = plt.figure(2)
plt.scatter(X_train, Y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="black", linewidth=2)
plt.title("Training set results")
plt.xlabel("Years of Experience")
plt.ylabel("Annual Salary")
plt2.show()

# Create a plot to visualise the test set results
plt3 = plt.figure(3)
plt.scatter(X_test, Y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="black", linewidth=2)
plt.title("Test set results")
plt.xlabel("Years of Experience")
plt.ylabel("Annual Salary")
plt3.show()
