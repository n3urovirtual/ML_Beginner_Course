# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Fit Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, Y)
Y_pred = regressor.predict(X)

# Create a plot to visualise Linear Regression results
plt1 = plt.figure(1)
plt.scatter(X, Y, c="black")
plt.plot(X, Y_pred, color="red")
plt.title("Linear Regression Results")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt1.show()

# Make a new prediction using Linear Regression
lin_prediction = regressor.predict([[8.5]])

"""3rd degree Polynomial"""
# Fit Polynomial Regression to the dataset (degree=3)
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=3)
X_polynomial = polynomial.fit_transform(X)
polynomial.fit(X_polynomial, Y)
regressor2 = LinearRegression()
regressor2.fit(X_polynomial, Y)

Y_pred2 = regressor2.predict(X_polynomial)

# Create a plot to visualise the Polynomial Regression results
plt2 = plt.figure(2)
plt.scatter(X, Y, color="black")
plt.plot(X, Y_pred2, color="green")
plt.title("Polynomial Regression Results")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt2.show()

# Make a new prediction using Polynomial Regression
poly_pred1 = regressor2.predict(polynomial.fit_transform([[8.5]]))

"""2nd degree Polynomial"""
# Fit Polynomial Regression to the dataset (degree=2)
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=2)
X_polynomial = polynomial.fit_transform(X)
polynomial.fit(X_polynomial, Y)
regressor3 = LinearRegression()
regressor3.fit(X_polynomial, Y)

Y_pred3 = regressor3.predict(X_polynomial)

# Create a plot to visualise the Polynomial Regression results
plt3 = plt.figure(3)
plt.scatter(X, Y, color="black")
plt.plot(X, Y_pred3, color="green")
plt.title("Polynomial Regression Results")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt3.show()

# Make a new prediction using Polynomial Regression
poly_pred2 = regressor3.predict(polynomial.fit_transform([[8.5]]))
